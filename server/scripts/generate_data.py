import pandas as pd
import random
import numpy as np

# --- CONFIGURATION ---
NUM_FACE_SHAPES = 5  # 0:Heart, 1:Oblong, 2:Oval, 3:Round, 4:Square
NUM_GLASSES = 50     # We simulate 50 pairs of glasses in our store
NUM_SAMPLES = 10000  # How many fake interactions to generate

# --- THE "TEACHER" RULES ---
# We define which glasses "technically" fit which face.
# The AI will discover these patterns by looking at the data.
# Range 0-9   -> Best for Heart
# Range 10-19 -> Best for Oblong
# Range 20-29 -> Best for Oval
# Range 30-39 -> Best for Round
# Range 40-49 -> Best for Square
PERFECT_MATCHES = {
    0: range(0, 10),   # Heart
    1: range(10, 20),  # Oblong
    2: range(20, 30),  # Oval
    3: range(30, 40),  # Round
    4: range(40, 50)   # Square
}

# Oval faces (ID 2) are "Universal", so we give them wider preferences
# They technically look good in almost anything
OVAL_BONUS_RANGE = range(0, 50)

def generate_synthetic_data():
    data = []

    print(f"Generating {NUM_SAMPLES} synthetic interactions...")

    for _ in range(NUM_SAMPLES):
        # 1. Pick a random user (Face Shape)
        user_id = random.randint(0, NUM_FACE_SHAPES - 1)
        
        # 2. Pick a random pair of glasses
        item_id = random.randint(0, NUM_GLASSES - 1)
        
        # 3. Determine: Did they LIKE it? (The Label)
        # Start with a base "No" (0)
        label = 0
        
        # RULE A: The Perfect Match
        # If the glasses are in their "Best For" range, high chance of Like
        if item_id in PERFECT_MATCHES[user_id]:
            # 85% chance they liked it (High signal)
            if random.random() < 0.85:
                label = 1
                
        # RULE B: The Oval Exception
        # Oval faces are lucky, they might like random stuff more often
        elif user_id == 2 and item_id in OVAL_BONUS_RANGE:
            # 60% chance they liked it
            if random.random() < 0.60:
                label = 1
                
        # RULE C: The "Noise" (Realism)
        # Sometimes people buy weird stuff that doesn't fit. 
        # We add 5% random likes to prevent the model from overfitting.
        elif random.random() < 0.05:
            label = 1

        data.append([user_id, item_id, label])

    # 4. Save to CSV
    df = pd.DataFrame(data, columns=['user_id', 'item_id', 'label'])
    
    # Shuffle the data so it's not ordered
    df = df.sample(frac=1).reset_index(drop=True)
    
    filename = "synthetic_interactions.csv"
    df.to_csv(filename, index=False)
    print(f"Success! Saved to '{filename}'")
    print(df.head())

if __name__ == "__main__":
    generate_synthetic_data()