import numpy as np
import string
import os
import csv
from datetime import datetime
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# =============================
# CONFIGURATION
# =============================

MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
MAX_SAMPLES = 800
TOP_K = 10

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]

RESULTS_DIR = "experiments/msmarco"
ABLATION_FILE = os.path.join(RESULTS_DIR, "ablation_results.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)

if not os.path.exists(ABLATION_FILE):
    with open(ABLATION_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "alpha",
            "use_stopwords",
            "use_normalization",
            "mean_ndcg",
            "mean_mrr",
            "num_queries"
        ])

STOPWORDS = set([
    "the", "a", "an", "is", "are", "of", "to", "and",
    "in", "on", "for", "with", "as", "by", "at", "from"
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

def tokenize(text, use_stopwords=True):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    if use_stopwords:
        return [t for t in tokens if t not in STOPWORDS]
    return tokens

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
all_embeddings_norm = model.encode(
    all_passages,
    normalize_embeddings=True,
    batch_size=64,
    show_progress_bar=True
)

all_embeddings_raw = model.encode(
    all_passages,
    normalize_embeddings=False,
    batch_size=64,
    show_progress_bar=False
)

# =============================
# ABLATION LOOP
# =============================

print("\nRunning ablation study...\n")

for use_stopwords in [True, False]:
    for use_norm in [True, False]:

        embeddings = all_embeddings_norm if use_norm else all_embeddings_raw

        for alpha in ALPHAS:

            ndcg_scores = []
            mrr_scores = []

            for q_idx in range(len(queries)):

                query = queries[q_idx]
                passages = candidate_sets[q_idx]
                relevant = relevant_indices[q_idx]

                # ----- BM25 -----
                tokenized_docs = [tokenize(p, use_stopwords) for p in passages]
                bm25 = BM25Okapi(tokenized_docs)
                tokenized_query = tokenize(query, use_stopwords)
                bm25_scores = bm25.get_scores(tokenized_query)

                # ----- Dense -----
                query_vec = model.encode(
                    query,
                    normalize_embeddings=use_norm
                )

                semantic_scores = []
                for p in passages:
                    idx = passage_to_idx[p]
                    semantic_scores.append(np.dot(embeddings[idx], query_vec))

                # ----- Hybrid -----
                hybrid_scores = [
                    alpha * semantic_scores[i] +
                    (1 - alpha) * (bm25_scores[i] / (bm25_scores[i] + 1) if bm25_scores[i] > 0 else 0)
                    for i in range(len(passages))
                ]

                ranked = list(np.argsort(hybrid_scores)[::-1])

                ndcg_scores.append(ndcg_at_k(relevant, ranked, TOP_K))
                mrr_scores.append(mrr_at_k(relevant, ranked, TOP_K))

            mean_ndcg = np.mean(ndcg_scores)
            mean_mrr = np.mean(mrr_scores)

            print(f"Alpha={alpha} | Stopwords={use_stopwords} | Norm={use_norm}")
            print(f"NDCG={mean_ndcg:.4f} | MRR={mean_mrr:.4f}")
            print("-" * 50)

            with open(ABLATION_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    alpha,
                    use_stopwords,
                    use_norm,
                    mean_ndcg,
                    mean_mrr,
                    len(queries)
                ])

print("\nAblation results saved to:")
print(ABLATION_FILE)
