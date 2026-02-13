import numpy as np
import time
import string
import os
import csv
from datetime import datetime
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# =====================================================
# CONFIG
# =====================================================

MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
MAX_SAMPLES = 800
ALPHA = 1.0
TOP_K = 10

RESULTS_DIR = "experiments/msmarco"
RESULTS_FILE = os.path.join(RESULTS_DIR, "latency_storage.csv")

STOPWORDS = set([
    "the", "a", "an", "is", "are", "of", "to", "and",
    "in", "on", "for", "with", "as", "by", "at", "from"
])

os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

print("Loading MS MARCO subset...")
dataset = load_dataset("ms_marco", "v1.1", split=f"train[:{MAX_SAMPLES}]")

queries = []
candidate_sets = []
relevant_indices = []

for row in dataset:
    query = row["query"]
    passages = row["passages"]["passage_text"]
    labels = row["passages"]["is_selected"]

    if sum(labels) == 1:
        queries.append(query)
        candidate_sets.append(passages)
        relevant_indices.append(labels.index(1))

print(f"Usable queries: {len(queries)}")

# =====================================================
# COLLECT UNIQUE PASSAGES
# =====================================================

all_passages = list({p for passages in candidate_sets for p in passages})
passage_to_idx = {p: i for i, p in enumerate(all_passages)}

print(f"Unique passages: {len(all_passages)}")

# =====================================================
# LOAD MODEL
# =====================================================

model = SentenceTransformer(MODEL_NAME)

print("Encoding all passages (float32)...")
all_embeddings = model.encode(
    all_passages,
    normalize_embeddings=True,
    batch_size=64,
    show_progress_bar=True
)

# =====================================================
# QUANTIZATION
# =====================================================

print("Quantizing embeddings to int8...")
emb_int8 = (all_embeddings * 127).astype(np.int8)
emb_int8_deq = emb_int8.astype(np.float32) / 127.0

# =====================================================
# STORAGE ANALYSIS
# =====================================================

float_size_bytes = all_embeddings.nbytes
int8_size_bytes = emb_int8.nbytes

compression_ratio = float_size_bytes / int8_size_bytes
space_saved_percent = (1 - int8_size_bytes / float_size_bytes) * 100

print("\nStorage Comparison:")
print(f"Float32 total size: {float_size_bytes / (1024**2):.2f} MB")
print(f"Int8 total size:    {int8_size_bytes / (1024**2):.2f} MB")
print(f"Compression ratio:  {compression_ratio:.2f}x")
print(f"Space saved:        {space_saved_percent:.2f}%")

# =====================================================
# TOKENIZER
# =====================================================

def tokenize(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS]

# =====================================================
# LATENCY PROFILING
# =====================================================

total_query_encode = 0
total_bm25_time = 0
total_semantic_float = 0
total_semantic_int8 = 0
total_sort_time = 0
total_pipeline_time = 0

print("\nRunning latency profiling...\n")

for q_idx in range(len(queries)):

    start_pipeline = time.perf_counter()

    query = queries[q_idx]
    passages = candidate_sets[q_idx]

    # ---------------------
    # Query Encoding
    # ---------------------
    start = time.perf_counter()
    query_vec = model.encode(query, normalize_embeddings=True)
    total_query_encode += time.perf_counter() - start

    # ---------------------
    # BM25
    # ---------------------
    start = time.perf_counter()
    tokenized_docs = [tokenize(p) for p in passages]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    total_bm25_time += time.perf_counter() - start

    # ---------------------
    # FLOAT semantic scoring
    # ---------------------
    start = time.perf_counter()
    semantic_float = []
    for p in passages:
        idx = passage_to_idx[p]
        semantic_float.append(np.dot(all_embeddings[idx], query_vec))
    total_semantic_float += time.perf_counter() - start

    # ---------------------
    # INT8 semantic scoring
    # ---------------------
    start = time.perf_counter()
    semantic_int8 = []
    for p in passages:
        idx = passage_to_idx[p]
        semantic_int8.append(np.dot(emb_int8_deq[idx], query_vec))
    total_semantic_int8 += time.perf_counter() - start

    # ---------------------
    # Sorting
    # ---------------------
    start = time.perf_counter()
    hybrid_scores = [
        ALPHA * semantic_float[i] +
        (1 - ALPHA) * (bm25_scores[i] / (bm25_scores[i] + 1) if bm25_scores[i] > 0 else 0)
        for i in range(len(passages))
    ]
    ranked = np.argsort(hybrid_scores)[::-1][:TOP_K]
    total_sort_time += time.perf_counter() - start

    total_pipeline_time += time.perf_counter() - start_pipeline

# =====================================================
# AVERAGES
# =====================================================

num_q = len(queries)

avg_query_encode = (total_query_encode / num_q) * 1000
avg_bm25 = (total_bm25_time / num_q) * 1000
avg_sem_float = (total_semantic_float / num_q) * 1000
avg_sem_int8 = (total_semantic_int8 / num_q) * 1000
avg_sort = (total_sort_time / num_q) * 1000
avg_pipeline = (total_pipeline_time / num_q) * 1000

print("\nAverage per-query latency (ms):\n")
print(f"Query encoding:      {avg_query_encode:.3f} ms")
print(f"BM25 stage:          {avg_bm25:.3f} ms")
print(f"Semantic float:      {avg_sem_float:.3f} ms")
print(f"Semantic int8:       {avg_sem_int8:.3f} ms")
print(f"Sorting stage:       {avg_sort:.3f} ms")
print(f"Full pipeline:       {avg_pipeline:.3f} ms")

# =====================================================
# LOG RESULTS
# =====================================================

if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "model",
            "num_queries",
            "num_passages",
            "query_encode_ms",
            "bm25_ms",
            "semantic_float_ms",
            "semantic_int8_ms",
            "sorting_ms",
            "full_pipeline_ms",
            "float_size_MB",
            "int8_size_MB",
            "compression_ratio",
            "space_saved_percent"
        ])

with open(RESULTS_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        datetime.now().isoformat(),
        MODEL_NAME,
        num_q,
        len(all_passages),
        avg_query_encode,
        avg_bm25,
        avg_sem_float,
        avg_sem_int8,
        avg_sort,
        avg_pipeline,
        float_size_bytes / (1024**2),
        int8_size_bytes / (1024**2),
        compression_ratio,
        space_saved_percent
    ])

print("\nResults logged to:")
print(RESULTS_FILE)
