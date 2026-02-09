import pandas as pd
import numpy as np
import random

# 1. SETTINGS
NUM_USERS = 100
NUM_ITEMS = 50
NUM_INTERACTIONS = 500

print("Generating synthetic data for Spectacular...")

# 2. GENERATE USERS (with Facial Features)
# We simulate the 5 geometric features mentioned in your system design
user_data = []
for user_id in range(NUM_USERS):
    # Random normalized values between 0.0 and 1.0
    cheekbone_jaw_ratio = round(random.uniform(0.5, 1.0), 2)
    face_height_width_ratio = round(random.uniform(0.5, 1.0), 2)
    jawline_angle = round(random.uniform(0.0, 1.0), 2)
    is_round = random.choice([0, 1])
    is_
    is_square = 1 - is_round # If not round, we'll say it's square for simplicity
    
    user_data.append([user_id, cheekbone_jaw_ratio, face_height_width_ratio, jawline_angle, is_round, is_square])

users_df = pd.DataFrame(user_data, columns=['user_id', 'ratio_cheek_jaw', 'ratio_height_width', 'jaw_angle', 'is_round', 'is_square'])
users_df.to_csv('users.csv', index=False)
print(f"✅ Created users.csv with {NUM_USERS} users and facial stats.")

# 3. GENERATE ITEMS (Glasses)
item_data = []
for item_id in range(NUM_ITEMS):
    style = random.choice(['Aviator', 'Wayfarer', 'Round', 'Square'])
    material = random.choice(['Metal', 'Plastic'])
    item_data.append([item_id, style, material])

items_df = pd.DataFrame(item_data, columns=['item_id', 'style', 'material'])
items_df.to_csv('items.csv', index=False)
print(f"✅ Created items.csv with {NUM_ITEMS} pairs of glasses.")

# 4. GENERATE INTERACTIONS (Likes/Clicks)
# This simulates users clicking on glasses they like
interaction_data = []
for _ in range(NUM_INTERACTIONS):
    u_id = random.randint(0, NUM_USERS - 1)
    i_id = random.randint(0, NUM_ITEMS - 1)
    rating = 1 # Implicit feedback (1 = liked/clicked)
    interaction_data.append([u_id, i_id, rating])

interactions_df = pd.DataFrame(interaction_data, columns=['user_id', 'item_id', 'interaction'])
interactions_df.to_csv('interactions.csv', index=False)
print(f"✅ Created interactions.csv with {NUM_INTERACTIONS} interactions.")