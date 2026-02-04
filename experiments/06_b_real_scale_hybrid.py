import time
import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

# 1. SETUP
MODEL_NAME = "prajjwal1/bert-tiny"
embedding_dim = 128
ALPHA = 0.8 # The Golden Ratio

print(f"Initializing {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

# 2. LOAD REAL DATA
print(f"Loading real AG News dataset (~120k unique docs)...")
dataset = load_dataset("ag_news", split="train") 
db_text = dataset['text'] 
num_docs = len(db_text)
print(f"Database Size: {num_docs} unique documents.")

# 3. INDEXING (FAISS)
print("\n--- Indexing Phase ---")
index = faiss.IndexFlatL2(embedding_dim) 

# Encode in batches
batch_size = 10000 
print(f"Encoding {num_docs} docs...")
start_index = time.time()

for i in range(0, num_docs, batch_size):
    batch = db_text[i : i + batch_size]
    vecs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(vecs)
    index.add(vecs)

print(f"Indexing Time: {time.time() - start_index:.2f}s")

# 4. HYBRID SEARCH FUNCTION
def hybrid_search(query):
    print(f"\nQUERY: '{query}'")
    
    # Step A: Vector Search (Top 100)
    q_vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)
    D, I = index.search(q_vec, k=100)
    
    # Step B: Re-Ranking
    candidates = []
    query_tokens = set(query.lower().split())
    
    for rank, doc_idx in enumerate(I[0]):
        vec_score = D[0][rank]
        
        # --- THE FIX IS HERE ---
        # We force convert numpy int64 to python int
        safe_idx = int(doc_idx) 
        doc_content = db_text[safe_idx]
        # -----------------------
        
        # Calculate Keyword Score
        doc_tokens = set(doc_content.lower().split())
        intersection = query_tokens.intersection(doc_tokens)
        key_score = len(intersection) / len(query_tokens)
        
        # Apply Golden Ratio
        final_score = (ALPHA * vec_score) + ((1-ALPHA) * key_score)
        
        candidates.append({
            "id": safe_idx,
            "snippet": doc_content[:100] + "...",
            "final_score": final_score,
            "vec_score": vec_score,
            "key_score": key_score
        })
    
    # Step C: Sort
    candidates.sort(key=lambda x: x['final_score'], reverse=True)
    
    # Step D: Display
    print(f"{'FINAL':<8} | {'VEC':<6} | {'KEY':<6} | {'SNIPPET'}")
    print("-" * 80)
    for c in candidates[:5]:
        print(f"{c['final_score']:.4f}   | {c['vec_score']:.4f} | {c['key_score']:.4f} | {c['snippet']}")

# 5. RUN THE TEST
hybrid_search("apple financial revenue")