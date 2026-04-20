import torch
import json
import numpy as np
import math
import os
import sys

# Ensure it can find the models folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.model import HybridNeuMF

# --- CONFIGURATION ---
DB_PATH = "../database/glasses_database.json" 
MODEL_PATH = "../spectacular_hybrid.pth"      
NUM_SHAPES = 5
TEST_USERS = 1000
K = 5 # For HR@5 and NDCG@5

def load_environment():
    """Loads the database and the trained model."""
    print("Loading database and model...")
    with open(DB_PATH, "r") as f:
        glasses_db = json.load(f)
    
    num_items = len(glasses_db)
    
    # Initialize model with 4 client features to match the engineered width_diff
    model = HybridNeuMF(
        num_face_shapes=NUM_SHAPES, 
        num_items=num_items, 
        num_client_features=4, # Updated from 3 to 4
        num_item_features=5,
        factor_num=32
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    
    return glasses_db, model, num_items

def test_rule_adherence(glasses_db, model, num_items):
    """
    Sanity Check: Verifies if Square Face (4) recommendations follow 
    the training logic (preferring shapes 0 and 1).
    """
    print("\n" + "="*50)
    print(" SANITY CHECK: RULE ADHERENCE (Square Face)")
    print("="*50)
    
    target_shape = 4 
    base_client_vec = [1.0, 1.0, 1.0] # [cheek_jaw, face_hw, midface]
    
    shape_tensor = torch.tensor([target_shape] * num_items)
    item_ids_tensor = torch.tensor(range(num_items))
    
    engineered_client_feats = []
    item_feats_list = []
    for i in range(num_items):
        g = glasses_db.get(str(i), {})
        g_width = g.get("width", 0.5)
        
        # FEATURE ENGINEERING: Calculate width_diff for every pair
        width_diff = abs(base_client_vec[1] - g_width)
        engineered_client_feats.append(base_client_vec + [width_diff])
        
        item_feats_list.append([
            g_width, g.get("height", 0.5), 
            g.get("bridge_pos", 0.5), g.get("normalized_material", 0.0), 
            g.get("normalized_rim", 0.0)
        ])
        
    client_feat_tensor = torch.tensor(engineered_client_feats, dtype=torch.float32)
    item_feat_tensor = torch.tensor(item_feats_list, dtype=torch.float32)

    with torch.no_grad():
        predictions = model(shape_tensor, item_ids_tensor, client_feat_tensor, item_feat_tensor)
    
    scored_items = []
    for i, score in enumerate(predictions.tolist()):
        meta = glasses_db.get(str(i), {})
        scored_items.append({
            "glass_id": i,
            "glass_shape_id": meta.get("shape_id", -1),
            "score": round(score, 4)
        })
        
    scored_items.sort(key=lambda x: x["score"], reverse=True)
    
    print("Top 5 Recommendations for SQUARE face (Expected frame shape: 1 or 0):")
    successes = 0
    for rank, item in enumerate(scored_items[:K]):
        match = "✅" if item["glass_shape_id"] in [0, 1] else "❌"
        if match == "✅": successes += 1
        print(f"Rank {rank+1}: Glass ID {item['glass_id']} | Frame Shape: {item['glass_shape_id']} | Score: {item['score']} {match}")
    
    print(f"\nRule Adherence Rate for Top 5: {successes}/{K} ({(successes/K)*100}%)")

def evaluate_metrics(glasses_db, model, num_items):
    """
    Evaluates Hit Ratio and NDCG by identifying ALL ground truth 
    best items for 1000 synthetic users.
    """
    print("\n" + "="*50)
    print(f" EVALUATING RANKING METRICS (HR@{K} & NDCG@{K})")
    print("="*50)
    
    hits = 0
    ndcg_sum = 0.0
    
    # Pre-build item features to calculate synthetic ground truth
    item_list = []
    for i in range(num_items):
        g = glasses_db.get(str(i), {})
        item_list.append({
            "id": i,
            "shape_id": g.get("shape_id", -1),
            "width": g.get("width", 0.5),
            "height": g.get("height", 0.5),
            "feats": [
                g.get("width", 0.5), g.get("height", 0.5), 
                g.get("bridge_pos", 0.5), g.get("normalized_material", 0.0), 
                g.get("normalized_rim", 0.0)
            ]
        })

    item_feat_tensor = torch.tensor([item["feats"] for item in item_list], dtype=torch.float32)
    item_ids_tensor = torch.tensor(range(num_items))

    for user in range(TEST_USERS):
        u_shape = np.random.randint(0, NUM_SHAPES)
        u_vec = [np.random.uniform(0.7, 1.3) for _ in range(3)]
        
        # 1. Calculate true scores for ALL items
        true_scores = []
        
        for item in item_list:
            score = 0.4
            if u_shape == 1 and item["height"] > 0.55: score += 0.4
            if u_shape == 3 and item["shape_id"] in [2, 4]: score += 0.4
            if u_shape == 0 and item["shape_id"] in [0, 4]: score += 0.4
            if u_shape == 2: score += 0.4
            if u_shape == 4:
                if item["shape_id"] in [1, 0]: score += 0.6
                elif item["shape_id"] == 2: score -= 0.4
            
            width_diff = abs(u_vec[1] - item["width"])
            if width_diff < 0.15: score += 0.3
            elif width_diff > 0.25: score -= 0.3
            
            true_scores.append((item["id"], score))
            
        # FIXED: Find the absolute highest score possible for this user
        max_true_score = max(s[1] for s in true_scores)
        
        # FIXED: Get ALL items that share this maximum score (the "perfect" matches)
        target_item_ids = [s[0] for s in true_scores if s[1] == max_true_score]
        
        # 2. Prepare engineered features for batch inference
        shape_tensor = torch.tensor([u_shape] * num_items)
        engineered_client_feats = []
        for item in item_list:
            diff = abs(u_vec[1] - item["width"])
            engineered_client_feats.append(u_vec + [diff])
            
        client_feat_tensor = torch.tensor(engineered_client_feats, dtype=torch.float32)
        
        # 3. Predict and Rank
        with torch.no_grad():
            predictions = model(shape_tensor, item_ids_tensor, client_feat_tensor, item_feat_tensor)
        
        ranked_indices = torch.argsort(-predictions).tolist()
        
        # 4. Calculate Metrics
        top_k_preds = ranked_indices[:K]
        
        # Hit Ratio: Did we recommend at least one perfect item?
        if any(t_id in top_k_preds for t_id in target_item_ids):
            hits += 1
            
        # NDCG: Find the highest rank of a perfect match
        best_rank = -1
        for rank, item_id in enumerate(top_k_preds):
            if item_id in target_item_ids:
                best_rank = rank
                break # Found the highest-ranked perfect match
                
        if best_rank != -1:
            ndcg_sum += math.log(2) / math.log(best_rank + 2)

        if (user + 1) % 200 == 0:
            print(f"Processed {user + 1}/{TEST_USERS} test users...")

    hr_k = hits / TEST_USERS
    ndcg_k = ndcg_sum / TEST_USERS
    
    print("\n--- FINAL RESULTS ---")
    print(f"Hit Ratio (HR@{K}): {hr_k:.4f} ({hr_k * 100:.2f}% accuracy)")
    print(f"NDCG@{K}: {ndcg_k:.4f}")

def test_robustness(glasses_db, model, num_items, noise_level=0.05):
    """
    Tests if the model still performs well when the user's face measurements 
    are slightly "noisy" or imperfect (simulating real-world camera inaccuracies).
    """
    print("\n" + "="*50)
    print(f" STRESS TEST: ROBUSTNESS TO NOISE (+/- {noise_level*100}%)")
    print("="*50)
    
    hits = 0
    
    # Pre-build items (Same as before)
    item_list = []
    for i in range(num_items):
        g = glasses_db.get(str(i), {})
        item_list.append({
            "id": i, "shape_id": g.get("shape_id", -1), "width": g.get("width", 0.5), "height": g.get("height", 0.5),
            "feats": [g.get("width", 0.5), g.get("height", 0.5), g.get("bridge_pos", 0.5), g.get("normalized_material", 0.0), g.get("normalized_rim", 0.0)]
        })
    item_feat_tensor = torch.tensor([item["feats"] for item in item_list], dtype=torch.float32)
    item_ids_tensor = torch.tensor(range(num_items))

    for user in range(TEST_USERS):
        u_shape = np.random.randint(0, NUM_SHAPES)
        # 1. Base clean vector to calculate the "True" perfect item
        clean_u_vec = [np.random.uniform(0.7, 1.3) for _ in range(3)]
        
        true_scores = []
        for item in item_list:
            score = 0.4
            if u_shape == 1 and item["height"] > 0.55: score += 0.4
            if u_shape == 3 and item["shape_id"] in [2, 4]: score += 0.4
            if u_shape == 0 and item["shape_id"] in [0, 4]: score += 0.4
            if u_shape == 2: score += 0.4
            if u_shape == 4:
                if item["shape_id"] in [1, 0]: score += 0.6
                elif item["shape_id"] == 2: score -= 0.4
            width_diff = abs(clean_u_vec[1] - item["width"])
            if width_diff < 0.15: score += 0.3
            elif width_diff > 0.25: score -= 0.3
            true_scores.append((item["id"], score))
            
        max_true_score = max(s[1] for s in true_scores)
        target_item_ids = [s[0] for s in true_scores if s[1] == max_true_score]
        
        # 2. NOISE INJECTION: Simulate a slightly inaccurate camera scan
        noisy_u_vec = [val + np.random.normal(0, noise_level) for val in clean_u_vec]
        
        shape_tensor = torch.tensor([u_shape] * num_items)
        engineered_client_feats = []
        for item in item_list:
            # The model has to predict using the NOISY data
            diff = abs(noisy_u_vec[1] - item["width"])
            engineered_client_feats.append(noisy_u_vec + [diff])
            
        client_feat_tensor = torch.tensor(engineered_client_feats, dtype=torch.float32)
        
        with torch.no_grad():
            predictions = model(shape_tensor, item_ids_tensor, client_feat_tensor, item_feat_tensor)
        
        top_k_preds = torch.argsort(-predictions).tolist()[:K]
        if any(t_id in top_k_preds for t_id in target_item_ids):
            hits += 1

    hr_k = hits / TEST_USERS
    print(f"Hit Ratio with {noise_level*100}% Noise: {hr_k:.4f} ({hr_k * 100:.2f}%)")
    if hr_k > 0.85:
        print("✅ Excellent! The model generalized the rules and didn't overfit to exact numbers.")
    else:
        print("❌ Model may be overfitting. It broke down when given slight variations.")


def test_catalog_coverage(glasses_db, model, num_items):
    """
    Checks if the model is recommending a healthy variety of glasses, 
    or if it is stuck recommending the exact same 5 pairs to everyone.
    """
    print("\n" + "="*50)
    print(" SANITY CHECK: CATALOG COVERAGE / DIVERSITY")
    print("="*50)
    
    recommended_items = set()
    
    item_feats_list = []
    for i in range(num_items):
        g = glasses_db.get(str(i), {})
        item_feats_list.append([g.get("width", 0.5), g.get("height", 0.5), g.get("bridge_pos", 0.5), g.get("normalized_material", 0.0), g.get("normalized_rim", 0.0)])
    item_feat_tensor = torch.tensor(item_feats_list, dtype=torch.float32)
    item_ids_tensor = torch.tensor(range(num_items))

    for user in range(TEST_USERS):
        u_shape = np.random.randint(0, NUM_SHAPES)
        u_vec = [np.random.uniform(0.7, 1.3) for _ in range(3)]
        
        shape_tensor = torch.tensor([u_shape] * num_items)
        engineered_client_feats = []
        for i, item_feats in enumerate(item_feats_list):
            diff = abs(u_vec[1] - item_feats[0]) # u_vec width - item width
            engineered_client_feats.append(u_vec + [diff])
            
        client_feat_tensor = torch.tensor(engineered_client_feats, dtype=torch.float32)
        
        with torch.no_grad():
            predictions = model(shape_tensor, item_ids_tensor, client_feat_tensor, item_feat_tensor)
        
        top_k_preds = torch.argsort(-predictions).tolist()[:K]
        
        # Add the recommended items to our unique set
        for item_id in top_k_preds:
            recommended_items.add(item_id)

    coverage_percentage = (len(recommended_items) / num_items) * 100
    print(f"Total Unique Items Recommended: {len(recommended_items)} out of {num_items}")
    print(f"Catalog Coverage: {coverage_percentage:.1f}%")
    
    if coverage_percentage > 50:
        print("✅ Healthy Diversity! The model uses a good variety of your catalog.")
    else:
        print("⚠️ Warning: Low Coverage. The model relies heavily on a small subset of glasses.")
if __name__ == "__main__":
    try:
        db, model, num_items = load_environment()
        test_rule_adherence(db, model, num_items)
        evaluate_metrics(db, model, num_items)
        
        # --- NEW TESTS ---
        test_robustness(db, model, num_items, noise_level=0.05)
        test_catalog_coverage(db, model, num_items)
        
    except Exception as e:
        print(f"Error running tests: {e}")