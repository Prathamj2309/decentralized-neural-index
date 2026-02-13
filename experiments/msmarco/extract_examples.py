import numpy as np
import string
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
MAX_SAMPLES = 800
TOP_K = 10

STOPWORDS = set([
    "the", "a", "an", "is", "are", "of", "to", "and",
    "in", "on", "for", "with", "as", "by", "at", "from"
])

def tokenize(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS]

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

all_passages = list({p for passages in candidate_sets for p in passages})
passage_to_idx = {p: i for i, p in enumerate(all_passages)}

model = SentenceTransformer(MODEL_NAME)
all_embeddings = model.encode(
    all_passages,
    normalize_embeddings=True,
    batch_size=64,
    show_progress_bar=True
)

print("\nSearching for failure cases...\n")

found = 0

for q_idx in range(len(queries)):

    query = queries[q_idx]
    passages = candidate_sets[q_idx]
    relevant = relevant_indices[q_idx]

    # BM25
    tokenized_docs = [tokenize(p) for p in passages]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranked = list(np.argsort(bm25_scores)[::-1])

    # Dense
    query_vec = model.encode(query, normalize_embeddings=True)
    semantic_scores = []
    for p in passages:
        idx = passage_to_idx[p]
        semantic_scores.append(np.dot(all_embeddings[idx], query_vec))

    dense_ranked = list(np.argsort(semantic_scores)[::-1])

    bm25_rank = bm25_ranked.index(relevant)
    dense_rank = dense_ranked.index(relevant)

    if bm25_rank - dense_rank >=5:

        print("====================================")
        print(f"Query: {query}")
        print(f"BM25 rank: {bm25_rank}")
        print(f"Dense rank: {dense_rank}")
        print("\nRelevant Passage:")
        print(passages[relevant][:400])
        print("====================================\n")

        found += 1

        if found == 3:
            break
