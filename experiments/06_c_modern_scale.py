%%writefile experiments/06_d_modern_scale.py
import time
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi

# 1. SETUP
MODEL_NAME = "prajjwal1/bert-tiny"
DATASET_NAME = "heegyu/news-category-dataset"

print(f"Initializing {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

# 2. LOAD MODERN DATA (2012-2022)
print(f"Loading {DATASET_NAME}...")
# This dataset has 'headline' and 'short_description'
dataset = load_dataset(DATASET_NAME, split="train") 

# We combine headline + description for better search context
print("Processing text columns...")
db_text = []
for row in dataset:
    # "iPhone 14 Released. Apple announced the new phone today..."
    full_text = f"{row['headline']}. {row['short_description']}"
    db_text.append(full_text)

num_docs = len(db_text)
print(f"Database Size: {num_docs} modern documents (2012-2022).")

# 3. BUILD THE HERO INDEX (BM25)
print(f"\n--- Building Hero Index (BM25) ---")
start_bm25 = time.time()

# Tokenize
tokenized_corpus = [doc.lower().split() for doc in db_text]
bm25 = BM25Okapi(tokenized_corpus)

print(f"BM25 Index Ready: {time.time() - start_bm25:.2f}s")

# 4. THE SEARCH FUNCTION (Hero -> Vector)
def modern_search(query):
    print(f"\nQUERY: '{query}'")
    
    # --- STEP 1: HERO FILTER (BM25) ---
    tokenized_query = query.lower().split()
    doc_scores = bm25.get_scores(tokenized_query)
    
    # Get Top 50 candidates
    top_n_indices = np.argpartition(doc_scores, -50)[-50:]
    
    candidates = []
    
    for idx in top_n_indices:
        if doc_scores[idx] > 0:
            safe_idx = int(idx)
            candidates.append(db_text[safe_idx])
            
    if len(candidates) == 0:
        print("No exact matches found.")
        return

    # --- STEP 2: VIBE CHECK (Vector Re-Rank) ---
    query_vec = model.encode(query, convert_to_tensor=True)
    candidate_vecs = model.encode(candidates, convert_to_tensor=True)
    
    sem_scores = util.cos_sim(query_vec, candidate_vecs)[0]
    
    # --- STEP 3: DISPLAY ---
    results = []
    for i in range(len(candidates)):
        score = sem_scores[i].item()
        text = candidates[i]
        results.append((score, text))
        
    results.sort(key=lambda x: x[0], reverse=True)
    
    print(f"{'SCORE':<8} | {'SNIPPET'}")
    print("-" * 80)
    for score, text in results[:5]:
        # Truncate for display
        print(f"{score:.4f}   | {text[:100]}...")

# 5. RUN MODERN TESTS
modern_search("elon musk twitter")
modern_search("covid vaccine pfizer")
modern_search("messi world cup")