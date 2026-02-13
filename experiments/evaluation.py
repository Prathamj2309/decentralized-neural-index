import numpy as np
import csv
import os
import string
import time
from datetime import datetime
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# =====================================================
# CONFIG
# =====================================================

MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
TOP_K = 10
ALPHAS = np.arange(0.0, 1.05, 0.1)
MAX_SAMPLES = 1200
RESULTS_DIR = "experiments/msmarco"
RESULTS_FILE = os.path.join(RESULTS_DIR, "results.csv")

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
# TOKENIZER
# =====================================================

def tokenize(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS]

# =====================================================
# METRICS
# =====================================================

def ndcg_at_k(ranked, rel_idx):
    dcg = 0
    for i, idx in enumerate(ranked):
        rel = 1 if idx == rel_idx else 0
        dcg += rel / np.log2(i + 2)
    return dcg  # ideal DCG = 1

def mrr_at_k(ranked, rel_idx):
    for i, idx in enumerate(ranked):
        if idx == rel_idx:
            return 1 / (i + 1)
    return 0

# =====================================================
# CREATE RESULTS CSV (if not exists)
# =====================================================

if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "model",
            "num_queries",
            "num_unique_passages",
            "alpha",
            "ndcg_float",
            "mrr_float",
            "ndcg_int8",
            "mrr_int8",
            "semantic_time_float",
            "semantic_time_int8"
        ])

# =====================================================
# EVALUATION
# =====================================================

print("\nRunning Alpha Sweep...\n")

for alpha in ALPHAS:

    total_ndcg_float = 0
    total_mrr_float = 0
    total_ndcg_int8 = 0
    total_mrr_int8 = 0

    semantic_time_float = 0
    semantic_time_int8 = 0

    for q_idx in range(len(queries)):

        query = queries[q_idx]
        passages = candidate_sets[q_idx]
        rel_local_idx = relevant_indices[q_idx]

        # -----------------------------
        # BM25 (lexical stage)
        # -----------------------------
        tokenized_docs = [tokenize(p) for p in passages]
        bm25 = BM25Okapi(tokenized_docs)
        tokenized_query = tokenize(query)
        bm25_scores = bm25.get_scores(tokenized_query)

        # -----------------------------
        # Encode query
        # -----------------------------
        query_vec = model.encode(query, normalize_embeddings=True)

        # -----------------------------
        # FLOAT semantic scoring
        # -----------------------------
        start_float = time.perf_counter()
        semantic_scores_float = []
        for p in passages:
            idx = passage_to_idx[p]
            semantic_scores_float.append(
                np.dot(all_embeddings[idx], query_vec)
            )
        semantic_time_float += time.perf_counter() - start_float

        # -----------------------------
        # INT8 semantic scoring
        # -----------------------------
        start_int8 = time.perf_counter()
        semantic_scores_int8 = []
        for p in passages:
            idx = passage_to_idx[p]
            semantic_scores_int8.append(
                np.dot(emb_int8_deq[idx], query_vec)
            )
        semantic_time_int8 += time.perf_counter() - start_int8

        hybrid_scores_float = []
        hybrid_scores_int8 = []

        for i in range(len(passages)):
            lexical = bm25_scores[i]
            lexical = lexical / (lexical + 1) if lexical > 0 else 0

            score_float = alpha * semantic_scores_float[i] + (1 - alpha) * lexical
            score_int8 = alpha * semantic_scores_int8[i] + (1 - alpha) * lexical

            hybrid_scores_float.append(score_float)
            hybrid_scores_int8.append(score_int8)

        ranked_float = np.argsort(hybrid_scores_float)[::-1][:TOP_K]
        ranked_int8 = np.argsort(hybrid_scores_int8)[::-1][:TOP_K]

        total_ndcg_float += ndcg_at_k(ranked_float, rel_local_idx)
        total_mrr_float += mrr_at_k(ranked_float, rel_local_idx)

        total_ndcg_int8 += ndcg_at_k(ranked_int8, rel_local_idx)
        total_mrr_int8 += mrr_at_k(ranked_int8, rel_local_idx)

    avg_ndcg_float = total_ndcg_float / len(queries)
    avg_mrr_float = total_mrr_float / len(queries)

    avg_ndcg_int8 = total_ndcg_int8 / len(queries)
    avg_mrr_int8 = total_mrr_int8 / len(queries)

    print(f"Alpha={alpha:.1f}")
    print(f"  Float32 → NDCG@10={avg_ndcg_float:.4f} | MRR@10={avg_mrr_float:.4f}")
    print(f"  Int8    → NDCG@10={avg_ndcg_int8:.4f} | MRR@10={avg_mrr_int8:.4f}")
    print("-" * 60)

    # Log results
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            MODEL_NAME,
            len(queries),
            len(all_passages),
            alpha,
            avg_ndcg_float,
            avg_mrr_float,
            avg_ndcg_int8,
            avg_mrr_int8,
            semantic_time_float,
            semantic_time_int8
        ])

print("\nExperiment complete. Results saved to:")
print(RESULTS_FILE)
