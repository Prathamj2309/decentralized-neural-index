import os
import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

# 1. SETUP THE ENGINE
# We use the winner of our "Audition"
MODEL_NAME = "prajjwal1/bert-tiny"
print(f"Initializing Search Engine with {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

def parse_vector_from_html(html_content):
    """
    Simulates the Crawler.
    It extracts ONLY the vector from the <meta> tag.
    It does NOT read the body text (Privacy/Speed feature).
    """
    # Regex to find: <meta name="neural-index" content="0.123, 0.456...">
    match = re.search(r'<meta name="neural-index" content="([^"]+)"', html_content)
    if match:
        vector_str = match.group(1)
        # Convert string "0.123, 0.456" back to float list
        return np.array([float(x) for x in vector_str.split(", ")], dtype=np.float32)
    return None

def create_fake_internet():
    """
    Simulates the Internet.
    We create 4 HTML files with PRE-COMPUTED tags.
    
    PROTOCOL UPDATE: 
    We now embed "{Title} + {Text}" to fix the context context issue.
    """
    os.makedirs("fake_internet", exist_ok=True)
    
    # The "Websites"
    # Note: We kept your edit to the finance page for maximum clarity.
    pages = {
        "page1_tech_apple.html": "The iPhone 15 release date has been announced by Apple Inc today.",
        "page2_fruit_recipes.html": "Apple pie recipes involve baking granny smith apples with cinnamon.",
        "page3_finance_revenue.html": "Apple's Wall Street revenue targets were missed by the tech giant this quarter.",
        "page4_irrelevant.html": "The quick brown fox jumps over the lazy dog."
    }
    
    print("\n--- Indexing the Web (Smart Mode: Title + Text) ---")
    for filename, text in pages.items():
        # 1. Extract a "Title" from the filename
        # "page3_finance.html" -> "finance"
        # In a real scenario, this would be the <title> tag.
        title_keyword = filename.replace("page", "").replace(".html", "").replace("_", " ").strip()
        
        # 2. Smart Input Strategy
        # We tell the model: "This is a FINANCE page about..."
        smart_content = f"{title_keyword}: {text}"
        
        # A. Webmaster generates vector
        raw_vec = model.encode(smart_content)
        
        # B. Webmaster quantizes it (The Protocol: 3 decimals)
        quantized = [round(x, 3) for x in raw_vec]
        vec_str = ", ".join(map(str, quantized))
        
        # C. Webmaster publishes HTML
        html = f"""
        <html>
            <head>
                <meta name="neural-index" content="{vec_str}">
                <title>{filename}</title>
            </head>
            <body>
                <p>{text}</p>
                </body>
        </html>
        """
        with open(f"fake_internet/{filename}", "w") as f:
            f.write(html)
        print(f"Indexed: {filename} (Category: '{title_keyword}')")

def search(query):
    print(f"\nSearching for: '{query}'")
    
    # 1. Embed Query
    query_vec = model.encode(query, convert_to_tensor=True)
    
    results = []
    for filename in os.listdir("fake_internet"):
        filepath = f"fake_internet/{filename}"
        with open(filepath, "r") as f:
            content = f.read()
            
        doc_vec = parse_vector_from_html(content)
        
        if doc_vec is not None:
            # A. Vector Score (The "Brain")
            doc_tensor = torch.tensor(doc_vec)
            sem_score = util.cos_sim(query_vec, doc_tensor)[0].item()
            
            # B. Keyword/Title Bonus (The "Tie-Breaker")
            # We check if words from the query appear in the filename (simulating the <title>)
            # simple logic: split query into words, check if they exist in filename
            bonus = 0.0
            query_words = query.lower().split()
            filename_lower = filename.lower()
            
            # If the filename contains a query word (e.g. "finance" or "tech"), boost it.
            matches = sum(1 for word in query_words if word in filename_lower)
            if matches > 0:
                bonus = 0.05 * matches  # 5% boost per word match
            
            final_score = sem_score + bonus
            
            # Store detail for debugging
            results.append((filename, final_score, sem_score, bonus))
            
    # 3. Rank
    results.sort(key=lambda x: x[1], reverse=True)
    
    # 4. Display
    for rank, (page, final, sem, bonus) in enumerate(results, 1):
        # We print the breakdown so you can see the hybrid logic working
        print(f"#{rank}: {page:<20} Score: {final:.4f} (Vec: {sem:.4f} + Bonus: {bonus:.2f})")

if __name__ == "__main__":
    # Step A: Build the fake websites
    create_fake_internet()
    
    # Step B: Run the "Apple" Test
    # Goal: Finance page should now beat Fruit page
    search("apple revenue")  
    
    # Step C: Run a Control Test
    search("baking ingredients")