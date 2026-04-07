import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim

class HybridNeuMF(pl.LightningModule):
    def __init__(self, num_face_shapes, num_items, num_client_features=4, num_item_features=5, factor_num=32, lr=0.005, epochs=30):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.epochs = epochs

        self.embed_shape_GMF = nn.Embedding(num_face_shapes, factor_num)
        self.embed_item_GMF = nn.Embedding(num_items, factor_num)
        self.embed_shape_MLP = nn.Embedding(num_face_shapes, factor_num)
        self.embed_item_MLP = nn.Embedding(num_items, factor_num)

        self.client_processor = nn.Sequential(
            nn.Linear(num_client_features, factor_num),
            nn.BatchNorm1d(factor_num),
            nn.ReLU()
        )
        self.item_processor = nn.Sequential(
            nn.Linear(num_item_features, factor_num),
            nn.BatchNorm1d(factor_num),
            nn.ReLU()
        )

        # Added Dropout to prevent overfitting on complex engineered continuous features
        self.mlp = nn.Sequential(
            nn.Linear(factor_num * 4, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # FIXED: Removed the overwriting layer. 
        # GMF outputs 32, MLP outputs 32 -> Total input is 64
        self.predict_layer = nn.Linear(factor_num + 32, 1)
        
        # FIXED: Swapped Binary Cross Entropy for Mean Squared Error
        self.criterion = nn.MSELoss() 
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.01)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, face_shape_id, item_id, client_features, item_features):
        s_gmf = self.embed_shape_GMF(face_shape_id)
        i_gmf = self.embed_item_GMF(item_id)
        out_gmf = s_gmf * i_gmf

        s_mlp = self.embed_shape_MLP(face_shape_id)
        i_mlp = self.embed_item_MLP(item_id)
        c_feat = self.client_processor(client_features)
        i_feat = self.item_processor(item_features)

        input_mlp = torch.cat([s_mlp, i_mlp, c_feat, i_feat], dim=-1)
        out_mlp = self.mlp(input_mlp)

        combined = torch.cat([out_gmf, out_mlp], dim=-1)
        
        # FIXED: Removed Sigmoid. We want the raw continuous regression score.
        return self.predict_layer(combined).squeeze()

    def training_step(self, batch, batch_idx):
        s, it, cf, itf, target = batch
        preds = self(s, it, cf, itf)
        loss = self.criterion(preds, target)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        # FIXED: T_max now accurately reflects your max_epochs to decay properly
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        return [optimizer], [scheduler]