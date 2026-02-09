import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from model.model import HybridNeuMF  # Importing your NEW model file

# --- CONFIG ---
BATCH_SIZE = 32
EPOCHS = 10
NUM_SHAPES = 5
NUM_ITEMS = 50

class HybridDataset(Dataset):
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.shapes = torch.tensor(self.df['shape_id'].values, dtype=torch.long)
        self.items = torch.tensor(self.df['item_id'].values, dtype=torch.long)
        # Combine the 3 feature columns into a single Float Tensor
        self.features = torch.tensor(self.df[['cheek_jaw', 'face_hw', 'midface']].values, dtype=torch.float32)
        self.labels = torch.tensor(self.df['label'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.shapes[idx], self.items[idx], self.features[idx], self.labels[idx]

# --- TRAIN ---
if __name__ == "__main__":
    print("1. Loading Data...")
    dataset = HybridDataset("./data/hybrid_interactions.csv")
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("2. Initializing HybridNeuMF...")
    # Initialize with 3 geometric features
    model = HybridNeuMF(NUM_SHAPES, NUM_ITEMS, num_geometric_features=3) 
    
    # Simple Training Loop (Manual for script simplicity, or use Trainer)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCELoss()
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for shapes, items, feats, labels in train_loader:
            optimizer.zero_grad()
            prediction = model(shapes, items, feats)
            loss = loss_fn(prediction, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss:.4f}")

    print("3. Saving Weights...")
    torch.save(model.state_dict(), "spectacular_hybrid.pth")
    print("✅ Model Saved: spectacular_hybrid.pth")