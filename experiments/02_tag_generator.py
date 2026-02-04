from sentence_transformers import SentenceTransformer, util
import torch

# 1. Load the Winner
MODEL_NAME = "prajjwal1/bert-tiny"
model = SentenceTransformer(MODEL_NAME)

def test_quantization_impact():
    # A. The Data
    query = "apple revenue"
    doc_tech = "The iPhone 15 was released by Apple Inc. last week."
    doc_fruit = "Apple pie is a delicious dessert made with cinnamon."
    
    # B. Generate RAW Vectors (High Precision)
    q_vec = model.encode(query, convert_to_tensor=True)
    d_vec_tech = model.encode(doc_tech, convert_to_tensor=True)
    
    # Baseline Score (The "Perfect" Score)
    raw_score = util.cos_sim(q_vec, d_vec_tech)[0].item()
    
    print(f"{'Precision':<15} | {'Size (Bytes)':<12} | {'Score':<12} | {'Quality Loss'}")
    print("-" * 65)
    
    # C. Test Rounding (The Protocol)
    # We test rounding to 5, 3, 2, and 1 decimal places
    for decimals in [5, 3, 2, 1]:
        # 1. Simulate the Webmaster creating the tag
        tech_list = d_vec_tech.tolist()
        rounded_list = [round(x, decimals) for x in tech_list]
        
        # 2. Measure Payload Size
        payload_string = ", ".join(map(str, rounded_list))
        size_bytes = len(payload_string)
        
        # 3. Measure if the Search Engine still understands it
        rounded_tensor = torch.tensor(rounded_list)
        new_score = util.cos_sim(q_vec, rounded_tensor)[0].item()
        
        loss = raw_score - new_score
        
        print(f"{decimals:<15} | {size_bytes:<12} | {new_score:.4f}       | {loss:.5f}")

if __name__ == "__main__":
    test_quantization_impact()