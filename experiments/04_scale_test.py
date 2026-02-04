import time
import torch
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

# 1. SETUP
MODEL_NAME = "prajjwal1/bert-tiny"
print(f"Loading {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

def quantize_and_store(vector):
    """Simulates the storage protocol (3 decimals)"""
    return np.array([round(x, 3) for x in vector], dtype=np.float32)

# 2. BUILD THE DATABASE (1,000 Real Articles)
print("Downloading 'AG News' dataset (Real World Data)...")
# We take the first 1000 entries from the test set
dataset = load_dataset("ag_news", split="test[:1000]")

# We need to simulate our "Metadata Strategy" (Title + Text)
# The dataset has 'text' and a label. We will use the label as a proxy for 'Title/Category'.
# Labels: 0=World, 1=Sports, 2=Business, 3=Sci/Tech
label_map = {0: "World News", 1: "Sports", 2: "Business Finance", 3: "Science Technology"}

print("\n--- Indexing 1,000 Documents... ---")
start_time = time.time()

index_vectors = []
index_metadata = []

for i, row in enumerate(dataset):
    category = label_map[row['label']]
    text = row['text']
    
    # HYBRID PROTOCOL: Embed "Category: Text"
    smart_content = f"{category}: {text}"
    
    # Generate & Quantize
    raw_vec = model.encode(smart_content)
    q_vec = quantize_and_store(raw_vec)
    
    index_vectors.append(q_vec)
    index_metadata.append({
        "id": i,
        "category": category,
        "snippet": text[:100] + "..." # Store snippet for display
    })

# Convert to Tensor for fast search
index_tensor = torch.tensor(np.array(index_vectors))
print(f"Indexing Complete in {time.time() - start_time:.2f}s")

# 3. THE SEARCH ENGINE
def run_scale_search(query):
    print(f"\nSearching 1,000 docs for: '{query}'")
    
    # A. Encode Query
    query_vec = model.encode(query, convert_to_tensor=True)
    
    # B. Vector Search (The "Brain")
    # This does 1,000 cosine calculations instantly
    sem_scores = util.cos_sim(query_vec, index_tensor)[0]
    
    # C. Hybrid Reranking (The "Bonus")
    final_results = []
    
    for idx, score in enumerate(sem_scores):
        meta = index_metadata[idx]
        vec_score = score.item()
        
        # KEYWORD BOOSTER
        # Does the category or text contain query terms?
        bonus = 0.0
        query_terms = query.lower().split()
        combined_text = (meta['category'] + " " + meta['snippet']).lower()
        
        matches = sum(1 for word in query_terms if word in combined_text)
        if matches > 0:
            bonus = 0.05 * matches
            
        final_score = vec_score + bonus
        final_results.append((final_score, meta))
        
    # D. Sort and Show Top 5
    final_results.sort(key=lambda x: x[0], reverse=True)
    
    print(f"{'SCORE':<8} | {'CATEGORY':<18} | {'SNIPPET'}")
    print("-" * 80)
    for score, meta in final_results[:5]:
        print(f"{score:.4f}   | {meta['category']:<18} | {meta['snippet']}")

# 4. RUN TESTS
# Can it find the specific "Apple" business news among 1,000 random stories?
# (Note: AG News contains actual articles about Apple, Microsoft, etc.)
run_scale_search("apple financial revenue")
run_scale_search("olympic gold medal")
run_scale_search("microsoft software release")