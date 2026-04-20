import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
import sys
import sqlite3

# Ensure models can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.model import HybridNeuMF

# --- OPTIMIZED HYPERPARAMETERS ---
NUM_SHAPES = 5
BATCH_SIZE = 256        
LEARNING_RATE = 0.001   
FACTOR_NUM = 32         
EPOCHS = 30             
SAMPLES_PER_EPOCH = 128 * 1000 
FEEDBACK_OVERSAMPLE = 20 # How many times to repeat real feedback per epoch so it isn't ignored

class GlassesDataset(Dataset):
    def __init__(self, size, glasses_db, real_feedback=[]):
        self.size = size
        self.glasses_db = glasses_db
        self.num_items = len(glasses_db)
        self.real_feedback = real_feedback
        
        # Total size is synthetic samples + oversampled real feedback
        self.total_size = self.size + len(self.real_feedback)

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # --- 1. REAL USER FEEDBACK PATH ---
        if idx < len(self.real_feedback):
            row = self.real_feedback[idx]
            item_idx = row['glass_id']
            u_shape = row['shape_id']
            u_vec = [row['cj'], row['hw'], row['mid']]
            liked = row['liked']
            
            # Fetch item features safely
            g = self.glasses_db.get(str(item_idx), {})
            g_vec = [
                g.get('width', 0.5), g.get('height', 0.5), 
                g.get('bridge_pos', 0.5), g.get('normalized_material', 0.0), 
                g.get('normalized_rim', 0.0)
            ]
            
            # Recreate engineered feature
            width_diff = abs(u_vec[1] - g_vec[0])
            u_vec_extended = u_vec + [width_diff]
            
            # Strong signals: 1.2 for liked, -0.2 for disliked
            final_label = 1.2 if liked else -0.2
            
            return (
                torch.tensor(u_shape, dtype=torch.long), 
                torch.tensor(item_idx, dtype=torch.long), 
                torch.tensor(u_vec_extended, dtype=torch.float32),
                torch.tensor(g_vec, dtype=torch.float32), 
                torch.tensor(final_label, dtype=torch.float32)
            )

        # --- 2. SYNTHETIC GENERATION PATH ---
        u_shape = np.random.randint(0, NUM_SHAPES)
        u_vec = [np.random.uniform(0.7, 1.3) for _ in range(3)]
        
        item_idx = np.random.randint(0, self.num_items)
        g = self.glasses_db[str(item_idx)]
        g_vec = [
            g.get('width', 0.5), g.get('height', 0.5), 
            g.get('bridge_pos', 0.5), g.get('normalized_material', 0.0), 
            g.get('normalized_rim', 0.0)
        ]
        
        score = 0.4 
        
        # Categorical Logic (Shape matching)
        if u_shape == 1 and g_vec[1] > 0.55: score += 0.4
        if u_shape == 3 and g.get('shape_id') in [2, 4]: score += 0.4
        if u_shape == 0 and g.get('shape_id') in [0, 4]: score += 0.4
        if u_shape == 2: score += 0.4
        if u_shape == 4:
            if g.get('shape_id') in [1, 0]: score += 0.6
            elif g.get('shape_id') == 2: score -= 0.4
        
        # Continuous Logic (Size matching)
        width_diff = abs(u_vec[1] - g_vec[0])
        if width_diff < 0.15: score += 0.3
        elif width_diff > 0.25: score -= 0.3

        u_vec_extended = u_vec + [width_diff]
        final_label = score 
        
        return (
            torch.tensor(u_shape, dtype=torch.long), 
            torch.tensor(item_idx, dtype=torch.long), 
            torch.tensor(u_vec_extended, dtype=torch.float32),
            torch.tensor(g_vec, dtype=torch.float32), 
            torch.tensor(final_label, dtype=torch.float32)
        )

def load_real_feedback():
    """Reads feedback.db and returns a list of dictionaries."""
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'feedback.db'))
    feedback_data = []
    
    if not os.path.exists(db_path):
        print("⚠️ No feedback.db found. Training on synthetic data only.")
        return feedback_data
        
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Ensure the table actually exists to prevent errors
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'")
        if not cursor.fetchone():
            print("⚠️ Feedback table does not exist yet. Training on synthetic data only.")
            return feedback_data

        cursor.execute("SELECT glass_id, detected_face_shape_id, cheek_jaw_ratio, face_hw_ratio, midface_ratio, liked FROM feedback")
        rows = cursor.fetchall()
        
        for row in rows:
            feedback_data.append({
                'glass_id': row[0],
                'shape_id': row[1],
                'cj': row[2] if row[2] else 1.0,
                'hw': row[3] if row[3] else 1.0,
                'mid': row[4] if row[4] else 1.0,
                'liked': bool(row[5])
            })
            
        conn.close()
        print(f"✅ Extracted {len(feedback_data)} unique real user interactions.")
        
        # Oversample real feedback so it has a strong impact
        if len(feedback_data) > 0:
            oversampled_data = feedback_data * FEEDBACK_OVERSAMPLE
            print(f"🔥 Oversampled to {len(oversampled_data)} interactions per epoch.")
            return oversampled_data
            
    except Exception as e:
        print(f"❌ Error loading feedback.db: {e}")
        
    return feedback_data

def main():
    # --- ROBUST DATABASE LOADING ---
    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'database', 'glasses_database.json'))
    
    try:
        with open(db_path, "r") as f:
            glasses_db = json.load(f)
        print(f"✅ Loaded {len(glasses_db)} glasses from catalog.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: Could not load catalog!\nReason: {e}")
        return 

    # --- LOAD REAL FEEDBACK ---
    real_feedback = load_real_feedback()

    # --- DATASET & DATALOADER ---
    dataset = GlassesDataset(SAMPLES_PER_EPOCH, glasses_db, real_feedback)
    
    # Note: Because DataLoader shuffles the data, real feedback and synthetic
    # data will be perfectly mixed together in the batches.
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True)

    # --- MODEL INITIALIZATION ---
    model = HybridNeuMF(
        num_face_shapes=NUM_SHAPES, 
        num_items=len(glasses_db), 
        factor_num=FACTOR_NUM,
        lr=LEARNING_RATE,
        epochs=EPOCHS
    )

    # --- TRAINER ---
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        gradient_clip_val=1.0,
        accelerator="auto",
        devices=1,
        precision="16-mixed", 
        enable_progress_bar=True
    )

    print(f"🚀 Starting Training for {EPOCHS} Epochs...")
    trainer.fit(model, train_loader)

    # --- SAVING WEIGHTS ---
    save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'spectacular_hybrid.pth'))
    torch.save(model.state_dict(), save_path)
    print(f"✅ Training Complete! Model saved to {save_path}")

if __name__ == "__main__":
    main()