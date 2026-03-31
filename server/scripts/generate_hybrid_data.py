import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
from models.model import HybridNeuMF 

with open("glasses_database.json", "r") as f:
    glasses_db = json.load(f)

NUM_ITEMS = len(glasses_db)
NUM_SHAPES = 5

def generate_training_batch(batch_size=32):
    shape_ids = []
    item_ids = []
    client_feats = []
    item_feats = []
    labels = []

    for _ in range(batch_size):
        u_shape = np.random.randint(0, NUM_SHAPES)
        u_feats = [np.random.uniform(0.7, 1.3) for _ in range(3)]
        
        i_id = np.random.randint(0, NUM_ITEMS)
        g = glasses_db[str(i_id)]
        
        i_feats = [g['width'], g['height'], g['bridge_pos'], g['normalized_material'], g['normalized_rim']]
        
        is_good_shape = (u_shape == 3 and g['shape_id'] == 2) or (u_shape == 4 and g['shape_id'] == 1)
        
        label = 1.0 if (is_good_shape or np.random.random() > 0.7) else 0.0
        
        shape_ids.append(u_shape)
        item_ids.append(i_id)
        client_feats.append(u_feats)
        item_feats.append(i_feats)
        labels.append(label)

    return (torch.tensor(shape_ids), torch.tensor(item_ids), 
            torch.tensor(client_feats, dtype=torch.float32), 
            torch.tensor(item_feats, dtype=torch.float32), 
            torch.tensor(labels, dtype=torch.float32))

model = HybridNeuMF(NUM_SHAPES, NUM_ITEMS)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

print("🚀 Starting Training...")
model.train()
for epoch in range(100): 
    optimizer.zero_grad()
    s, i, cf, itf, target = generate_training_batch(128)
    
    output = model(s, i, cf, itf)
    loss = criterion(output, target)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")


torch.save(model.state_dict(), "spectacular_hybrid.pth")
print("✅ Weights saved to spectacular_hybrid.pth")