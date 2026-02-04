import time
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi

# 1. SETUP - THE UPGRADE
# old: MODEL_NAME = "prajjwal1/bert-tiny" (14MB)
# new: MODEL_NAME = "all-MiniLM-L6-v2" (80MB - The Gold Standard)
MODEL_NAME = "all-MiniLM-L6-v2" 
DATASET_NAME = "heegyu/news-category-dataset"

print(f"Initializing {MODEL_NAME} (The Smarter Brain)...")
model = SentenceTransformer(MODEL_NAME)

# 2. LOAD DATA
print(f"Loading modern news dataset...")
dataset = load_dataset(DATASET_NAME, split="train") 
db_text = []
for row in dataset:
    full_text = f"{row['headline']}. {row['short_description']}"
    db_text.append(full_text)

# 3. BUILD HERO INDEX (BM25)
print(f"\n--- Building Hero Index (BM25) ---")
tokenized_corpus = [doc.lower().split() for doc in db_text]
bm25 = BM25Okapi(tokenized_corpus)
print("BM25 Ready.")

# 4. SEARCH FUNCTION
def smart_search(query):
    print(f"\nQUERY: '{query}'")
    
    # Step A: Hero Filter (BM25)
    tokenized_query = query.lower().split()
    doc_scores = bm25.get_scores(tokenized_query)
    top_n_indices = np.argpartition(doc_scores, -50)[-50:]
    
    candidates = []
    for idx in top_n_indices:
        if doc_scores[idx] > 0:
            candidates.append(db_text[int(idx)])
            
    if not candidates:
        print("No matches.")
        return

    # Step B: Vibe Check (Smarter Vectors)
    query_vec = model.encode(query, convert_to_tensor=True)
    candidate_vecs = model.encode(candidates, convert_to_tensor=True)
    sem_scores = util.cos_sim(query_vec, candidate_vecs)[0]
    
    # Step C: Display
    results = []
    for i in range(len(candidates)):
        results.append((sem_scores[i].item(), candidates[i]))
        
    results.sort(key=lambda x: x[0], reverse=True)
    
    print(f"{'SCORE':<8} | {'SNIPPET'}")
    print("-" * 80)
    for score, text in results[:5]:
        print(f"{score:.4f}   | {text[:100]}...")

# 5. RUN TESTS
smart_search("elon musk twitter")
smart_search("covid vaccine pfizer")
smart_search("messi world cup")