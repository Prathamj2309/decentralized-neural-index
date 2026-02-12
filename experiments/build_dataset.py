# build_dataset.py
import json
from datasets import load_dataset

def format_ds(ds, col_headline, col_body):
    return [f"{row[col_headline]}. {row[col_body]}"[:500] for row in ds]

print("Loading dataset...")

db_text = []

# Use smaller scale for research clarity (30–50k is enough)
ds1 = load_dataset("heegyu/news-category-dataset", split="train[:30000]")
db_text.extend(format_ds(ds1, 'headline', 'short_description'))

print(f"Total documents: {len(db_text)}")

# Save text corpus
with open("corpus.json", "w") as f:
    json.dump(db_text, f)

print("Saved corpus.json")
