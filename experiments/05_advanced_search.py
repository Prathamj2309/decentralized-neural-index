import time
import numpy as np
import torch
import random
from datetime import datetime, timedelta
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from rank_bm25 import BM25Okapi

# --- CONFIGURATION ---
RETRIEVAL_TOP_K = 50   # How many to fetch in Stage 1
RERANK_TOP_K = 5       # Final results to show
# We use a specialized Cross-Encoder (Tiny but sharp)
CROSS_MODEL_NAME = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
BI_MODEL_NAME = "prajjwal1/bert-tiny"

print(f"Loading Models...\nBi-Encoder: {BI_MODEL_NAME}\nCross-Encoder: {CROSS_MODEL_NAME}")
bi_encoder = SentenceTransformer(BI_MODEL_NAME)
cross_encoder = CrossEncoder(CROSS_MODEL_NAME)

# --- SIMULATING TIME ---
def generate_fake_dates(num_docs):
    """
    Assigns random dates within the last 30 days.
    Some are 'Today', some are 'Last Month'.
    """
    base_date = datetime.now()
    dates = []
    for _ in range(num_docs):
        days_old = random.randint(0, 30)
        date = base_date - timedelta(days=days_old)
        dates.append(date)
    return dates

def calculate_recency_boost(doc_date):
    """
    Decay Function: Gaussian Decay.
    Recent docs get score ~1.0. Old docs decay towards 0.5 or lower.
    """
    days_diff = (datetime.now() - doc_date).days
    # Logic: Score drops by half every 7 days (sigma=7)
    # Formula: exp(- (days^2) / (2 * sigma^2))
    # We add a floor (0.8) so old relevant content isn't completely killed.
    decay = np.exp(-(days_diff**2) / (2 * 100)) # 100 is variance control
    return 0.8 + (0.2 * decay) # Boost factor between 1.0 (Today) and 0.8 (Old)

# --- 1. INDEXING (THE HYBRID DATABASE) ---
print("\n--- Building Index (1,000 Docs) ---")
dataset = load_dataset("ag_news", split="test[:1000]")
fake_dates = generate_fake_dates(1000)

corpus_metadata = []
corpus_text = []

# A. Prepare Data
for i, row in enumerate(dataset):
    text = row['text']
    date = fake_dates[i]
    
    corpus_metadata.append({
        "id": i,
        "text": text,
        "date": date,
        "date_str": date.strftime("%Y-%m-%d")
    })
    corpus_text.append(text)

# B. Vector Indexing (Bi-Encoder)
print("1. Generating Vectors...")
corpus_embeddings = bi_encoder.encode(corpus_text, convert_to_tensor=True)

# C. BM25 Indexing (Keyword)
print("2. Building BM25 Index...")
tokenized_corpus = [doc.split(" ") for doc in corpus_text]
bm25 = BM25Okapi(tokenized_corpus)

# --- 2. SEARCH LOGIC ---

def advanced_search(query, user_needs_recency=False):
    print(f"\nQUERY: '{query}' (Recency Bias: {user_needs_recency})")
    
    # --- STAGE 1: HYBRID RETRIEVAL (RRF) ---
    
    # A. Vector Search
    query_vec = bi_encoder.encode(query, convert_to_tensor=True)
    sem_scores = util.cos_sim(query_vec, corpus_embeddings)[0]
    # Get Top-K indices
    vec_top_k = torch.topk(sem_scores, k=RETRIEVAL_TOP_K).indices.tolist()
    
    # B. BM25 Search
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    # Get Top-K indices (numpy argpartition is fast)
    bm25_top_k = np.argpartition(bm25_scores, -RETRIEVAL_TOP_K)[-RETRIEVAL_TOP_K:]
    
    # C. Reciprocal Rank Fusion (RRF)
    # We map "Doc ID" -> "RRF Score"
    rrf_scores = {}
    k = 60 # RRF constant
    
    for rank, doc_id in enumerate(vec_top_k):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k + rank))
        
    for rank, doc_id in enumerate(bm25_top_k):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1 / (k + rank))
    
    # Sort by RRF score and get top candidates for Re-Ranking
    candidates_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:RETRIEVAL_TOP_K]
    
    # --- STAGE 2: CROSS-ENCODER RE-RANKING ---
    
    # Prepare pairs for the Cross-Encoder: [[Query, Doc1], [Query, Doc2]...]
    cross_input = [[query, corpus_metadata[doc_id]['text']] for doc_id in candidates_ids]
    
    cross_scores = cross_encoder.predict(cross_input)
    
    # --- STAGE 3: TEMPORAL RE-SCORING ---
    
    final_results = []
    for i, doc_id in enumerate(candidates_ids):
        base_score = cross_scores[i] # This is the "Relevance" (0 to 1)
        doc_data = corpus_metadata[doc_id]
        
        # Apply Recency Bias if requested
        time_boost = 1.0
        if user_needs_recency:
            time_boost = calculate_recency_boost(doc_data['date'])
            
        final_score = base_score * time_boost
        final_results.append((doc_id, final_score, base_score, doc_data['date_str']))

    # Sort Final
    final_results.sort(key=lambda x: x[1], reverse=True)
    
    # --- DISPLAY ---
    print(f"{'SCORE':<8} | {'BASE':<6} | {'DATE':<12} | {'SNIPPET'}")
    print("-" * 90)
    for doc_id, final, base, date in final_results[:RERANK_TOP_K]:
        snippet = corpus_metadata[doc_id]['text'][:80].replace("\n", " ") + "..."
        print(f"{final:.4f}   | {base:.4f} | {date:<12} | {snippet}")

# --- 3. RUN EXPERIMENTS ---

# Test 1: Ambiguous Query (Requires Hybrid RRF to find the entity)
advanced_search("apple financial revenue")

# Test 2: Recency Test (Requires Temporal Bias)
# We expect the scores to change based on the random dates assigned
advanced_search("latest sports news", user_needs_recency=True)