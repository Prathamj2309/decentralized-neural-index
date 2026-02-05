import time
import string
import psutil
import os
import numpy as np
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi

# 1. SETUP
# We use the QA-Tuned Brain (22MB quantized size equivalent)
MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
print(f"Initializing {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

# 2. BUILD THE "FRANKENSTEIN" DATASET (Approaching 1 Million Docs)
print("\n--- ASSEMBLING 1 MILLION DOCS ---")

# Helper to format text consistently
def format_ds(ds, col_headline, col_body):
    return [f"{row[col_headline]}. {row[col_body]}"[:500] for row in ds]

db_text = []

# A. Load HuffPost (200k)
print("Loading HuffPost (200k)...")
ds1 = load_dataset("heegyu/news-category-dataset", split="train")
db_text.extend(format_ds(ds1, 'headline', 'short_description'))

# B. Load AG News (120k)
print("Loading AG News (120k)...")
ds2 = load_dataset("ag_news", split="train")
# AG News has 'text' column only, treating as body
db_text.extend([row['text'][:500] for row in ds2])

# C. Load XSum (BBC) (226k)
print("Loading XSum (BBC News) (226k)...")
ds3 = load_dataset("xsum", split="train")
db_text.extend(format_ds(ds3, 'id', 'document')) # XSum uses ID as a sort of header usually, using document

# D. Load CNN/DailyMail (300k) - Taking a slice to save download time if needed
print("Loading CNN/DailyMail (300k)...")
# version 3.0.0 is standard
ds4 = load_dataset("cnn_dailymail", "3.0.0", split="train")
db_text.extend(format_ds(ds4, 'article', 'highlights'))

total_docs = len(db_text)
print(f"\nTOTAL DATABASE SIZE: {total_docs} DISTINCT DOCUMENTS")
print(f"RAM Usage: {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.2f} MB")

# 3. BUILD KEYWORD INDEX (The Heavy Lift)
print(f"\n--- INDEXING {total_docs} DOCS ---")
start_bm25 = time.time()

def clean_tokenize(text):
    # Fast punctuation removal
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return text.split()

# This is the memory stress test
tokenized_corpus = [clean_tokenize(doc) for doc in db_text]
bm25 = BM25Okapi(tokenized_corpus)

print(f"BM25 Index Time: {time.time() - start_bm25:.2f}s")

# 4. HYBRID SEARCH (The Latency Test)
def hybrid_search(query):
    print(f"\n--- QUERY: '{query}' ---")
    start_time = time.time()
    
    # A. KEYWORD FILTER (Scan 1 Million Docs)
    tokenized_query = clean_tokenize(query)
    doc_scores = bm25.get_scores(tokenized_query)
    
    # Get Top 100 (Filtering Phase)
    # argpartition is O(n), very fast even for 1M
    top_n_indices = np.argpartition(doc_scores, -100)[-100:]
    
    candidates = []
    candidates_indices = []
    
    for idx in top_n_indices:
        if doc_scores[idx] > 0:
            candidates.append(db_text[int(idx)])
            candidates_indices.append(int(idx))
            
    filter_time = time.time()
    
    if not candidates:
        print("No keyword matches.")
        return

    # B. VECTOR RE-RANK (The Intelligence)
    # Only encode ~100 docs
    query_vec = model.encode(query, convert_to_tensor=True)
    cand_vecs = model.encode(candidates, convert_to_tensor=True)
    
    vector_scores = util.cos_sim(query_vec, cand_vecs)[0].cpu().numpy()
    
    # C. SCORE & SORT
    results = []
    for i in range(len(candidates)):
        # Normalize BM25 (Approximation)
        k_score = doc_scores[candidates_indices[i]] / 10.0
        v_score = vector_scores[i]
        
        # 70/30 Split
        final_score = (v_score * 0.7) + (k_score * 0.3)
        
        results.append({
            "score": final_score,
            "text": candidates[i],
            "src": "Unknown" # Tracking source would be cool but keep it simple
        })
        
    results.sort(key=lambda x: x['score'], reverse=True)
    
    end_time = time.time()
    
    print(f"Total Time: {(end_time - start_time)*1000:.2f}ms")
    print(f"(Filter: {(filter_time - start_time)*1000:.2f}ms | Vector: {(end_time - filter_time)*1000:.2f}ms)")
    
    print(f"{'SCORE':<8} | {'SNIPPET'}")
    print("-" * 80)
    for r in results[:5]:
        print(f"{r['score']:.4f}   | {r['text'][:100]}...")

# 5. THE SCALE TESTS
hybrid_search("cupertino tech giant revenue")
hybrid_search("elon musk twitter deal")
hybrid_search("messi world cup")