import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl
import logging

logger = logging.getLogger(__name__)

class GRUModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # GRU-Layer
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully Connected Layer für die Vorhersage
        self.fc = nn.Linear(hidden_size, 1)
        
        # Aktivierungsfunktion für binäre Klassifikation
        self.sigmoid = nn.Sigmoid()
        
        # Loss-Funktion
        self.loss_fn = nn.BCELoss()
        
        # Learning Rate
        self.lr = lr
        
        logger.info(f"GRU-Modell initialisiert: input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}")
    
    def forward(self, x):
        # x: [batch_size, seq_len, input_size]
        
        # GRU-Verarbeitung
        out, _ = self.gru(x)
        
        # Nimm nur den letzten Zeitpunkt der Sequenz
        out = out[:, -1, :]
        
        # Fully Connected Layer
        out = self.fc(out)
        
        # Sigmoid für binäre Klassifikation
        out = self.sigmoid(out)
        
        return out
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        # Logging
        self.log("train_loss", loss, prog_bar=True)
        
        # Berechne Accuracy
        preds = (y_hat > 0.5).float()
        acc = (preds == y).float().mean()
        self.log("train_acc", acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        # Logging
        self.log("val_loss", loss, prog_bar=True)
        
        # Berechne Accuracy
        preds = (y_hat > 0.5).float()
        acc = (preds == y).float().mean()
        self.log("val_acc", acc, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        # Logging
        self.log("test_loss", loss, prog_bar=True)
        
        # Berechne Accuracy
        preds = (y_hat > 0.5).float()
        acc = (preds == y).float().mean()
        self.log("test_acc", acc, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }