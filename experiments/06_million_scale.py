import time
import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# 1. SETUP
MODEL_NAME = "prajjwal1/bert-tiny"
embedding_dim = 128  # bert-tiny output size
num_docs = 1_000_000 # The Goal

print(f"Initializing {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

# 2. GENERATE FAKE MILLION DATASET
# Since AG News is small (120k), we will duplicate it to hit 1M.
print(f"Loading Dataset to simulate {num_docs} documents...")
dataset = load_dataset("ag_news", split="train") # 120k rows
real_count = len(dataset)

# We need a list of text snippets to act as our "Database"
# We will store only the text to save RAM, not the full object
db_text = []

print("Generating 1M strings (This might take a moment)...")
# We loop through the dataset multiple times to fill 1M slots
for i in range(num_docs):
    # Modulo operator wraps around the dataset
    original_text = dataset[i % real_count]['text']
    # We add a unique ID so they aren't EXACT duplicates
    text = f"DocID_{i}: {original_text}" 
    db_text.append(text)

print(f"Database Ready: {len(db_text)} documents.")

# 3. ENCODE & INDEX (THE HEAVY LIFTING)
# In production, this happens once and is saved to disk.
print("\n--- Indexing Phase ---")
start_index = time.time()

# FAISS requires vectors to be in float32 format
# We process in batches to avoid OOM (Out of Memory)
batch_size = 10000 
index = faiss.IndexFlatL2(embedding_dim) # Brute force L2 (Exact Search)
# Note: For massive speedups, use 'IndexIVFFlat', but 'IndexFlatL2' is accurate baseline.

print(f"Encoding in batches of {batch_size}...")
for i in range(0, num_docs, batch_size):
    batch_text = db_text[i : i + batch_size]
    
    # Encode
    batch_vectors = model.encode(batch_text, convert_to_numpy=True, show_progress_bar=False)
    
    # Add to FAISS Index
    faiss.normalize_L2(batch_vectors) # Essential for Cosine Similarity
    index.add(batch_vectors)
    
    if i % 100000 == 0:
        print(f"Indexed {i} / {num_docs}...")

end_index = time.time()
print(f"Indexing Complete! Time: {end_index - start_index:.2f}s")
print(f"Index Size: {index.ntotal} vectors")

# 4. THE MILLION-SCALE SEARCH
def search_million(query, k=5):
    print(f"\nSearching 1,000,000 docs for: '{query}'")
    start_search = time.time()
    
    # A. Encode Query
    query_vec = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    
    # B. Search Index
    # D = Distances (Scores), I = Indices (Doc IDs)
    D, I = index.search(query_vec, k)
    
    search_time = (time.time() - start_search) * 1000 # ms
    
    # C. Display
    print(f"Search Time: {search_time:.2f} ms")
    print(f"{'SCORE':<8} | {'SNIPPET'}")
    print("-" * 80)
    
    for rank, doc_idx in enumerate(I[0]):
        score = D[0][rank]
        # Retrieve text from our list
        snippet = db_text[doc_idx][:100] + "..."
        print(f"{score:.4f}   | {snippet}")

# 5. RUN TESTS
# Can it find the needle in a massive haystack?
search_million("apple financial revenue")
search_million("gold medal olympics")
search_million("new microsoft update")