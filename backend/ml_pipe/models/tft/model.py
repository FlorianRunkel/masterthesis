import torch
import torch.nn as nn
import pytorch_lightning as pl

class TFTModel(pl.LightningModule):
    def __init__(self, sequence_features=2, hidden_size=64, dropout=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.position_embedding = nn.Embedding(1000, 32)
        self.classifier = nn.Sequential(
            nn.Linear(sequence_features - 1 + 32, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 4)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x_seq):
        positions = x_seq[:, 0].long()
        position_embeddings = self.position_embedding(positions)
        wechselzeitraum = x_seq[:, 1].float().unsqueeze(-1)
        combined_features = torch.cat([position_embeddings, wechselzeitraum], dim=-1)
        return self.classifier(combined_features)

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")
        
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")
    
    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")
    
    def _shared_step(self, batch, stage):
        x_seq, y = batch
        y_hat = self(x_seq)
        loss = self.loss_fn(y_hat, y)
        
        # Metriken
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        
        # Logging
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-3,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }
