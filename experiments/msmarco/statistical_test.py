import numpy as np
import string
import os
import csv
from datetime import datetime
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from scipy.stats import ttest_rel, wilcoxon

# =============================
# CONFIGURATION
# =============================

MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
MAX_SAMPLES = 800
ALPHA = 1.0
TOP_K = 10

RESULTS_DIR = "experiments/msmarco"
STAT_FILE = os.path.join(RESULTS_DIR, "statistical_results.csv")

STOPWORDS = set([
    "the", "a", "an", "is", "are", "of", "to", "and",
    "in", "on", "for", "with", "as", "by", "at", "from"
])

os.makedirs(RESULTS_DIR, exist_ok=True)

# Create CSV header if not exists
if not os.path.exists(STAT_FILE):
    with open(STAT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "metric",
            "system_A",
            "system_B",
            "mean_A",
            "mean_B",
            "t_stat",
            "p_ttest",
            "wilcoxon_stat",
            "p_wilcoxon",
            "num_queries"
        ])

# =============================
# METRICS
# =============================

def ndcg_at_k(relevant_index, ranked_indices, k=10):
    if relevant_index in ranked_indices[:k]:
        rank = ranked_indices.index(relevant_index)
        return 1 / np.log2(rank + 2)
    return 0.0

def mrr_at_k(relevant_index, ranked_indices, k=10):
    if relevant_index in ranked_indices[:k]:
        rank = ranked_indices.index(relevant_index)
        return 1 / (rank + 1)
    return 0.0

def tokenize(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS]

# =============================
# LOAD DATA
# =============================

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

# =============================
# PRECOMPUTE EMBEDDINGS
# =============================

all_passages = list({p for passages in candidate_sets for p in passages})
passage_to_idx = {p: i for i, p in enumerate(all_passages)}

model = SentenceTransformer(MODEL_NAME)

print("Encoding passages...")
all_embeddings = model.encode(
    all_passages,
    normalize_embeddings=True,
    batch_size=64,
    show_progress_bar=True
)

# Quantization
emb_int8 = (all_embeddings * 127).astype(np.int8)
emb_int8_deq = emb_int8.astype(np.float32) / 127.0

# =============================
# COLLECT PER-QUERY METRICS
# =============================

ndcg_bm25 = []
ndcg_dense = []
ndcg_hybrid = []
ndcg_int8 = []

mrr_bm25 = []
mrr_dense = []
mrr_hybrid = []
mrr_int8 = []

print("\nEvaluating per-query scores...\n")

for q_idx in range(len(queries)):

    query = queries[q_idx]
    passages = candidate_sets[q_idx]
    relevant = relevant_indices[q_idx]

    # ----- BM25 -----
    tokenized_docs = [tokenize(p) for p in passages]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranked = list(np.argsort(bm25_scores)[::-1])

    # ----- Dense -----
    query_vec = model.encode(query, normalize_embeddings=True)

    semantic_float = []
    semantic_int8 = []

    for p in passages:
        idx = passage_to_idx[p]
        semantic_float.append(np.dot(all_embeddings[idx], query_vec))
        semantic_int8.append(np.dot(emb_int8_deq[idx], query_vec))

    dense_ranked = list(np.argsort(semantic_float)[::-1])
    int8_ranked = list(np.argsort(semantic_int8)[::-1])

    # ----- Hybrid -----
    hybrid_scores = [
        ALPHA * semantic_float[i] +
        (1 - ALPHA) * (bm25_scores[i] / (bm25_scores[i] + 1) if bm25_scores[i] > 0 else 0)
        for i in range(len(passages))
    ]

    hybrid_ranked = list(np.argsort(hybrid_scores)[::-1])

    # ----- Metrics -----
    ndcg_bm25.append(ndcg_at_k(relevant, bm25_ranked, TOP_K))
    ndcg_dense.append(ndcg_at_k(relevant, dense_ranked, TOP_K))
    ndcg_hybrid.append(ndcg_at_k(relevant, hybrid_ranked, TOP_K))
    ndcg_int8.append(ndcg_at_k(relevant, int8_ranked, TOP_K))

    mrr_bm25.append(mrr_at_k(relevant, bm25_ranked, TOP_K))
    mrr_dense.append(mrr_at_k(relevant, dense_ranked, TOP_K))
    mrr_hybrid.append(mrr_at_k(relevant, hybrid_ranked, TOP_K))
    mrr_int8.append(mrr_at_k(relevant, int8_ranked, TOP_K))

# =============================
# STATISTICAL TEST FUNCTION
# =============================

def run_tests(metric_name, name_a, scores_a, name_b, scores_b):

    t_stat, p_t = ttest_rel(scores_a, scores_b)
    w_stat, p_w = wilcoxon(scores_a, scores_b)

    mean_a = np.mean(scores_a)
    mean_b = np.mean(scores_b)

    print(f"\n{name_a} vs {name_b} ({metric_name})")
    print(f"Mean A: {mean_a:.4f}")
    print(f"Mean B: {mean_b:.4f}")
    print(f"Paired t-test: t={t_stat:.4f}, p={p_t:.6f}")
    print(f"Wilcoxon: W={w_stat:.4f}, p={p_w:.6f}")

    with open(STAT_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            metric_name,
            name_a,
            name_b,
            mean_a,
            mean_b,
            t_stat,
            p_t,
            w_stat,
            p_w,
            len(scores_a)
        ])

# =============================
# RUN TESTS
# =============================

print("\n==============================")
print("STATISTICAL SIGNIFICANCE TESTS")
print("==============================")

# NDCG
run_tests("NDCG", "Dense", ndcg_dense, "BM25", ndcg_bm25)
run_tests("NDCG", "Hybrid", ndcg_hybrid, "BM25", ndcg_bm25)
run_tests("NDCG", "Float", ndcg_dense, "Int8", ndcg_int8)

# MRR
run_tests("MRR", "Dense", mrr_dense, "BM25", mrr_bm25)
run_tests("MRR", "Hybrid", mrr_hybrid, "BM25", mrr_bm25)
run_tests("MRR", "Float", mrr_dense, "Int8", mrr_int8)

print("\nResults saved to:")
print(STAT_FILE)
