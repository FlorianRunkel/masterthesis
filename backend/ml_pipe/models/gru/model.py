import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, gru_output):
        # gru_output shape: (batch, seq_len, hidden_size)
        attention_weights = F.softmax(self.attention(gru_output), dim=1)
        # Weighted sum of GRU outputs
        context = torch.sum(attention_weights * gru_output, dim=1)
        return context, attention_weights

class GRUModel(pl.LightningModule):
    def __init__(self, seq_input_size=13, hidden_size=256, num_layers=2, dropout=0.2, lr=0.001):
        super().__init__()
        self.save_hyperparameters()

        # Bidirektionales GRU
        self.gru = nn.GRU(
            input_size=seq_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Bidirektional für besseres Sequenzverständnis
        )

        # Aufmerksamkeitsmechanismus
        self.attention = AttentionLayer(hidden_size * 2)  # *2 wegen bidirektional

        # Batch Normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)

        # Verbesserte Fully Connected Schichten
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 1)
        )

        # Kombinierte Loss-Funktion: MSE + L1 für robustere Vorhersagen
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.lr = lr

        # Metriken für Training und Validierung
        self.train_mse = nn.MSELoss()
        self.val_mse = nn.MSELoss()
        self.test_mse = nn.MSELoss()

    def forward(self, x):
        # GRU Verarbeitung
        gru_out, _ = self.gru(x)
        
        # Aufmerksamkeitsmechanismus
        context, attention_weights = self.attention(gru_out)
        
        # Batch Normalization
        context = self.batch_norm(context)
        
        # Fully Connected Schichten
        out = self.fc(context)
        
        return out, attention_weights

    def _shared_step(self, batch, stage):
        x, y = batch
        y_hat, _ = self(x)
        
        # Kombinierte Loss-Funktion für Training
        if stage == "train":
            mse_loss = self.mse_loss(y_hat, y)
            l1_loss = self.l1_loss(y_hat, y)
            loss = mse_loss + 0.1 * l1_loss
        else:
            # Nur MSE für Validierung und Test
            loss = self.mse_loss(y_hat, y)
        
        # Logging
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_mse", self.mse_loss(y_hat, y), prog_bar=True)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=0.01  # L2 Regularisierung
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }