import torch
import torch.nn as nn
import pytorch_lightning as pl

class HybridNeuMF(pl.LightningModule):
    def __init__(self, num_face_shapes, num_items, num_geometric_features=3, factor_num=8):
        """
        num_face_shapes: 5 (Heart, Square, etc.)
        num_items: Number of glasses in DB
        num_geometric_features: Dimensionality of the vector from client (e.g., 3 ratios)
        """
        super().__init__()

        # 1. Embeddings
        # We replace "User ID" with "Face Shape Embedding"
        self.embed_shape_GMF = nn.Embedding(num_face_shapes, factor_num)
        self.embed_item_GMF = nn.Embedding(num_items, factor_num)

        self.embed_shape_MLP = nn.Embedding(num_face_shapes, factor_num)
        self.embed_item_MLP = nn.Embedding(num_items, factor_num)

        # 2. Feature Processing Layer
        # Processes the raw ratios (e.g., 1.2, 0.8) into a dense representation
        self.feature_processor = nn.Linear(num_geometric_features, factor_num)

        # 3. MLP Part
        # Input: Shape_Embed + Item_Embed + Processed_Features
        mlp_input_size = (factor_num * 2) + factor_num 
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        # 4. Prediction Layer
        self.predict_layer = nn.Linear(factor_num + 8, 1) # GMF + MLP output
        self.sig = nn.Sigmoid()

    def forward(self, face_shape_id, item_id, geometric_features):
        # --- GMF Branch (Shape x Item) ---
        shape_embed_gmf = self.embed_shape_GMF(face_shape_id)
        item_embed_gmf = self.embed_item_GMF(item_id)
        output_gmf = shape_embed_gmf * item_embed_gmf

        # --- MLP Branch (Shape + Item + Features) ---
        shape_embed_mlp = self.embed_shape_MLP(face_shape_id)
        item_embed_mlp = self.embed_item_MLP(item_id)
        
        # Process the client-side features
        feature_embed = self.feature_processor(geometric_features)

        # Concatenate all signals
        input_mlp = torch.cat((shape_embed_mlp, item_embed_mlp, feature_embed), -1)
        output_mlp = self.mlp(input_mlp)

        # --- Fusion ---
        concat = torch.cat((output_gmf, output_mlp), -1)
        prediction = self.sig(self.predict_layer(concat))
        
        return prediction.squeeze()