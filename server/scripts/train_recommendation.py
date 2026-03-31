import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.model import HybridNeuMF

NUM_SHAPES = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.005
FACTOR_NUM = 32 
EPOCHS = 15
SAMPLES_PER_EPOCH = 128 * 1000 

class GlassesDataset(Dataset):
    def __init__(self, size, glasses_db):
        self.size = size
        self.glasses_db = glasses_db
        self.num_items = len(glasses_db)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
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
        if u_shape == 1 and g_vec[1] > 0.55:
            score += 0.4
        if u_shape == 3 and g.get('shape_id') in [2, 4]:
            score += 0.4
        if u_shape == 0 and g.get('shape_id') in [0, 4]:
            score += 0.4
        if u_shape == 2:
            score += 0.4
        if u_shape == 4:
            if g.get('shape_id') in [1, 0]:
                score += 0.6
            elif g.get('shape_id') == 2:
                score -= 0.4
        
        width_diff = abs(u_vec[1] - g_vec[0])
        if width_diff < 0.15:
            score += 0.3
        elif width_diff > 0.25:
            score -= 0.3
        
        final_label = max(0.05, min(0.95, score))
        
        return (
            torch.tensor(u_shape, dtype=torch.long), 
            torch.tensor(item_idx, dtype=torch.long), 
            torch.tensor(u_vec, dtype=torch.float32), 
            torch.tensor(g_vec, dtype=torch.float32), 
            torch.tensor(final_label, dtype=torch.float32)
        )

def main():
    db_path = os.path.join(os.path.dirname(__file__), '..', 'glasses_database.json')
    try:
        with open(db_path, "r") as f:
            glasses_db = json.load(f)
    except Exception as e:
        return

    dataset = GlassesDataset(SAMPLES_PER_EPOCH, glasses_db)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True)

    model = HybridNeuMF(
        num_face_shapes=NUM_SHAPES, 
        num_items=len(glasses_db), 
        factor_num=FACTOR_NUM,
        lr=LEARNING_RATE
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        gradient_clip_val=1.0,
        accelerator="auto",
        devices=1,
        precision="16-mixed"
    )

    trainer.fit(model, train_loader)

    save_path = os.path.join(os.path.dirname(__file__), '..', 'spectacular_hybrid.pth')
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    main()  