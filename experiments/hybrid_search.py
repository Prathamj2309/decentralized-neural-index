# hybrid_search.py

import json
import string
import numpy as np
from time import perf_counter
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# =========================
# CONFIGURATION
# =========================

MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
TOP_K = 100
ALPHA = 0.7  # Hybrid weight (semantic vs lexical)

STOPWORDS = set([
    "the", "a", "an", "is", "are", "of", "to", "and",
    "in", "on", "for", "with", "as", "by", "at", "from"
])

# =========================
# LOAD DATA
# =========================

print("Loading corpus...")
with open("corpus.json") as f:
    db_text = json.load(f)

print("Loading embeddings...")
emb_float = np.load("embeddings_float32.npy")
emb_int8 = np.load("embeddings_int8.npy")

# Dequantize int8 embeddings
emb_int8 = emb_int8.astype(np.float32) / 127.0

print(f"Float32 storage (MB): {emb_float.nbytes / (1024**2):.2f}")
print(f"Int8 storage (MB): {emb_int8.nbytes / (1024**2):.2f}")

# =========================
# BUILD BM25 INDEX
# =========================

def clean_tokenize(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS]

print("Building BM25 index...")
tokenized_corpus = [clean_tokenize(doc) for doc in db_text]
bm25 = BM25Okapi(tokenized_corpus)

# Load model (query embedding only)
model = SentenceTransformer(MODEL_NAME)

# =========================
# SEARCH FUNCTION
# =========================

def search(query):
    print("\n" + "="*60)
    print(f"QUERY: {query}")

    start_total = perf_counter()

    # ---- BM25 FILTER ----
    tokenized_query = clean_tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)

    top_indices = np.argpartition(bm25_scores, -TOP_K)[-TOP_K:]
    filter_time = perf_counter()

    # ---- QUERY EMBEDDING ----
    query_vec = model.encode(query, normalize_embeddings=True)

    # ---- FLOAT32 SEMANTIC RERANK ----
    start_float = perf_counter()
    float_scores = np.dot(emb_float[top_indices], query_vec)
    float_time = perf_counter() - start_float

    # ---- INT8 SEMANTIC RERANK ----
    start_int8 = perf_counter()
    int8_scores = np.dot(emb_int8[top_indices], query_vec)
    int8_time = perf_counter() - start_int8

    total_time = perf_counter() - start_total

    print(f"Filter time: {(filter_time - start_total)*1000:.2f} ms")
    print(f"Float32 rerank: {float_time*1000:.2f} ms")
    print(f"Int8 rerank: {int8_time*1000:.2f} ms")
    print(f"Total time: {total_time*1000:.2f} ms")

    # ---- HYBRID SCORING ----
    results = []

    for i, idx in enumerate(top_indices):
        bm = bm25_scores[idx]
        lexical_score = bm / (bm + 1) if bm > 0 else 0
        semantic_score = float_scores[i]

        final_score = ALPHA * semantic_score + (1 - ALPHA) * lexical_score

        results.append((idx, final_score))

    results.sort(key=lambda x: x[1], reverse=True)

    print("\nTop Results:")
    for idx, score in results[:5]:
        print(f"{score:.4f} | {db_text[idx][:100]}...")

# =========================
# RUN TEST QUERIES
# =========================

if __name__ == "__main__":
    search("cupertino tech giant revenue")
    search("elon musk twitter deal")
    search("messi world cup")
