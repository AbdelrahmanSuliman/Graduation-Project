import pandas as pd
import numpy as np
import random

# --- CONFIGURATION ---
NUM_USERS = 500   # More users to show patterns
NUM_ITEMS = 50    # 50 Glasses
INTERACTIONS = 5000

# Face Shape Mapping (Matches your API)
FACE_MAP = {0: "Heart", 1: "Oblong", 2: "Oval", 3: "Round", 4: "Square"}

def get_realistic_features(shape_id):
    """
    Generates geometric ratios based on the academic definitions 
    of face shapes (Reference: Report Section 4.1).
    """
    # Base jitter to make it realistic
    jitter = np.random.normal(0, 0.05)
    
    if shape_id == 0: # Heart: Wide cheeks, narrow jaw
        cheek_jaw = 1.45 + jitter
        face_hw = 1.2 + jitter
        midface = 0.5 + jitter
    elif shape_id == 1: # Oblong: Long face
        cheek_jaw = 1.1 + jitter
        face_hw = 1.55 + jitter # High height-to-width
        midface = 0.6 + jitter
    elif shape_id == 2: # Oval: Balanced
        cheek_jaw = 1.25 + jitter
        face_hw = 1.35 + jitter
        midface = 0.5 + jitter
    elif shape_id == 3: # Round: Short, wide
        cheek_jaw = 1.1 + jitter
        face_hw = 1.05 + jitter # 1:1 ratio
        midface = 0.45 + jitter
    elif shape_id == 4: # Square: Strong jaw
        cheek_jaw = 1.05 + jitter # Jaw width ~= Cheek width
        face_hw = 1.1 + jitter
        midface = 0.5 + jitter
        
    return cheek_jaw, face_hw, midface

# --- GENERATE DATA ---
data = []

for _ in range(INTERACTIONS):
    # 1. Pick a Random User Profile
    user_id = random.randint(0, NUM_USERS - 1)
    shape_id = random.randint(0, 4)
    
    # 2. Generate Biometric Features for this shape
    c_j, f_hw, mid = get_realistic_features(shape_id)
    
    # 3. Pick a Glass
    item_id = random.randint(0, NUM_ITEMS - 1)
    
    # 4. Logic: Does this glass fit this shape? (Synthetic Ground Truth)
    # Example Rule: Square (ID 4) likes Round Glasses (ID 30-39)
    label = 0
    if shape_id == 4 and 30 <= item_id < 40: label = 1
    elif shape_id == 0 and 0 <= item_id < 10: label = 1
    elif shape_id == 3 and 30 <= item_id < 40: label = 1 # Round likes Square/Rect frames
    elif shape_id == 1 and 10 <= item_id < 20: label = 1
    elif shape_id == 2: label = 1 if random.random() > 0.3 else 0 # Oval fits most
    
    data.append([user_id, shape_id, item_id, c_j, f_hw, mid, label])

# Save
df = pd.DataFrame(data, columns=[
    "user_id", "shape_id", "item_id", 
    "cheek_jaw", "face_hw", "midface", "label"
])
df.to_csv("hybrid_interactions.csv", index=False)
print("✅ Generated hybrid_interactions.csv with biometric data.")