import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import string
from time import perf_counter

# ========================
# CONFIG
# ========================

MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
TOP_K = 10
ALPHAS = np.arange(0.0, 1.05, 0.1)

STOPWORDS = set([
    "the", "a", "an", "is", "are", "of", "to", "and",
    "in", "on", "for", "with", "as", "by", "at", "from"
])

# ========================
# LOAD AG NEWS
# ========================

print("Loading AG News...")
dataset = load_dataset("ag_news", split="train[:20000]")

texts = [row["text"] for row in dataset]
labels = [row["label"] for row in dataset]

print("Encoding documents...")
model = SentenceTransformer(MODEL_NAME)

doc_embeddings = model.encode(
    texts,
    normalize_embeddings=True,
    batch_size=64,
    show_progress_bar=True
)

# ========================
# BM25
# ========================

def tokenize(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS]

tokenized_corpus = [tokenize(t) for t in texts]
bm25 = BM25Okapi(tokenized_corpus)

# ========================
# METRICS
# ========================

def ndcg_at_k(ranked_indices, query_label):
    dcg = 0
    for i, idx in enumerate(ranked_indices):
        rel = 1 if labels[idx] == query_label else 0
        dcg += rel / np.log2(i + 2)

    ideal_rels = sorted(
        [1 if labels[idx] == query_label else 0 for idx in ranked_indices],
        reverse=True
    )

    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0


def mrr_at_k(ranked_indices, query_label):
    for i, idx in enumerate(ranked_indices):
        if labels[idx] == query_label:
            return 1 / (i + 1)
    return 0

# ========================
# CATEGORY QUERIES
# ========================

category_queries = {
    0: "world politics government",
    1: "sports football soccer basketball",
    2: "business economy market finance",
    3: "technology science innovation ai"
}

print("\nRunning Alpha Sweep...\n")

for alpha in ALPHAS:

    total_ndcg = 0
    total_mrr = 0

    for label_id, query in category_queries.items():

        tokenized_query = tokenize(query)
        bm25_scores = bm25.get_scores(tokenized_query)

        query_vec = model.encode(query, normalize_embeddings=True)
        semantic_scores = np.dot(doc_embeddings, query_vec)

        hybrid_scores = []

        for i in range(len(texts)):
            lexical = bm25_scores[i]
            lexical = lexical / (lexical + 1) if lexical > 0 else 0
            semantic = semantic_scores[i]
            score = alpha * semantic + (1 - alpha) * lexical
            hybrid_scores.append(score)

        ranked = np.argsort(hybrid_scores)[::-1][:TOP_K]

        total_ndcg += ndcg_at_k(ranked, label_id)
        total_mrr += mrr_at_k(ranked, label_id)

    avg_ndcg = total_ndcg / len(category_queries)
    avg_mrr = total_mrr / len(category_queries)

    print(f"Alpha={alpha:.1f} | NDCG@10={avg_ndcg:.4f} | MRR@10={avg_mrr:.4f}")
