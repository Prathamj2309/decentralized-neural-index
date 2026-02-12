# build_embeddings.py
import json
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "multi-qa-MiniLM-L6-cos-v1"
model = SentenceTransformer(MODEL_NAME)

print("Loading corpus...")
with open("corpus.json") as f:
    db_text = json.load(f)

print("Encoding documents (this may take time)...")

doc_embeddings = model.encode(
    db_text,
    batch_size=64,
    normalize_embeddings=True,
    show_progress_bar=True
)

doc_embeddings = np.array(doc_embeddings, dtype=np.float32)

np.save("embeddings_float32.npy", doc_embeddings)
print("Saved float32 embeddings.")

# Quantize to int8
doc_embeddings_int8 = (doc_embeddings * 127).astype(np.int8)
np.save("embeddings_int8.npy", doc_embeddings_int8)
print("Saved int8 embeddings.")
