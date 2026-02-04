import time
import os
from sentence_transformers import SentenceTransformer, util

# --- ROUND 2 CANDIDATES ---
models_to_test = [
    "prajjwal1/bert-tiny",                    # The previous winner (Baseline)
    "TaylorAI/bge-micro-v2",                  # The Specialist (Distilled for Search)
    "sentence-transformers/paraphrase-MiniLM-L3-v2", # The "Middle Ground"
    "sentence-transformers/all-MiniLM-L6-v2"  # The Standard (for reference)
]

# --- THE TEST (UNCHANGED) ---
query = "apple revenue"
candidates = [
    "The iPhone 15 was released by Apple Inc. last week.",   # MATCH
    "Apple pie is a delicious dessert made with cinnamon.",  # MISMATCH
    "Microsoft announced new cloud computing earnings."      # RELATED (Tech)
]

print(f"{'Model Name':<50} | {'Time':<8} | {'Match':<8} | {'Mismatch':<8} | {'Gap'}")
print("-" * 100)

for model_name in models_to_test:
    try:
        start = time.time()
        model = SentenceTransformer(model_name)
        load_time = time.time() - start
        
        query_vec = model.encode(query, convert_to_tensor=True)
        doc_vecs = model.encode(candidates, convert_to_tensor=True)
        
        scores = util.cos_sim(query_vec, doc_vecs)[0]
        
        match = scores[0].item()
        mismatch = scores[1].item()
        gap = match - mismatch  # Bigger gap = Better model
        
        print(f"{model_name:<50} | {load_time:.2f}s   | {match:.4f}   | {mismatch:.4f}   | {gap:.4f}")
        
        del model
    except Exception as e:
        print(f"{model_name:<50} | FAILED TO LOAD: {e}")