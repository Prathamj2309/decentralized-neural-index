import numpy as np
from sentence_transformers import SentenceTransformer, util

# 1. SETUP THE LAB
model_name = "prajjwal1/bert-tiny"
print(f"Loading {model_name}...")
model = SentenceTransformer(model_name)

# The "Problem" Case from your 1M test
query = "apple financial revenue"

# The Candidates
documents = [
    # The Semantic Trap (High Vector Score, Low Keyword Match)
    # This tricked the model because it screams "Money" and "Tech"
    "Cisco Systems Inc. on Tuesday reported significant Q1 net income revenue growth.",
    
    # The Correct Answer (Should have higher Keyword Match)
    "Apple Inc. today announced financial results for its fiscal 2024 fourth quarter revenue."
]

doc_names = ["Cisco (Competitor)", "Apple (Target)"]

print(f"\n--- Tuning Hybrid Weights for: '{query}' ---")

# 2. CALCULATE RAW SCORES

# A. Vector Scores (The "Brain") - Cosine Similarity
query_vec = model.encode(query)
doc_vecs = model.encode(documents)
vector_scores = util.cos_sim(query_vec, doc_vecs)[0].tolist()

# B. Keyword Scores (The "Anchor")
# We simulate a simple overlap score (similar to BM25)
# Score = (Number of matching words) / (Length of Query)
keyword_scores = []
query_tokens = set(query.lower().split())

for doc in documents:
    doc_tokens = set(doc.lower().replace(".", "").split())
    # Count how many query words (apple, financial, revenue) appear in doc
    intersection = query_tokens.intersection(doc_tokens)
    score = len(intersection) / len(query_tokens)
    keyword_scores.append(score)

print(f"\nRAW METRICS:")
print(f"{'METRIC':<15} | {'CISCO':<10} | {'APPLE':<10}")
print("-" * 45)
print(f"{'Vector Score':<15} | {vector_scores[0]:.4f}     | {vector_scores[1]:.4f}")
print(f"{'Keyword Score':<15} | {keyword_scores[0]:.4f}     | {keyword_scores[1]:.4f}")

# 3. THE ALPHA SWEEP (Finding the Golden Ratio)
print(f"\n--- SWEEPING ALPHA (Weighting) ---")
print("Formula: Score = (Alpha * Vector) + ((1-Alpha) * Keyword)")
print("Alpha 1.0 = Pure Vector | Alpha 0.0 = Pure Keyword")
print("-" * 65)
print(f"{'ALPHA':<6} | {'CISCO SCORE':<12} | {'APPLE SCORE':<12} | {'WINNER':<10}")
print("-" * 65)

# We test values from 1.0 down to 0.0
for alpha in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
    # Calculate weighted scores
    score_cisco = (alpha * vector_scores[0]) + ((1-alpha) * keyword_scores[0])
    score_apple = (alpha * vector_scores[1]) + ((1-alpha) * keyword_scores[1])
    
    if score_apple > score_cisco:
        winner = "APPLE"
    else:
        winner = "CISCO (Fail)"
        
    print(f"{alpha:<6} | {score_cisco:.4f}       | {score_apple:.4f}       | {winner}")

print("\nCONCLUSION:")
print("Look for the highest Alpha where Apple still wins.")
print("High Alpha = Smarter (better synonyms). Low Alpha = Stricter (exact words).")