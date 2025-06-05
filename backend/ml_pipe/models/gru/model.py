import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics.regression import R2Score, MeanAbsolutePercentageError

'''
Attention Layer for GRU Model
'''
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

'''
GRU Model
'''
class GRUModel(pl.LightningModule):
    def __init__(self, seq_input_size=10, hidden_size=128, num_layers=4, dropout=0.2, lr=0.0003):
        super().__init__()
        self.save_hyperparameters()

        self.gru = nn.GRU(
            input_size=seq_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        self.attention = AttentionLayer(hidden_size * 2)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.r2 = R2Score()
        self.mape = MeanAbsolutePercentageError()
        self.lr = lr

    def forward(self, x):
        gru_out, _ = self.gru(x)
        attn_out, attn_weights = self.attention(gru_out)
        out = self.fc(attn_out)
        return out, attn_weights

    def _shared_step(self, batch, stage):
        x, y = batch
        y_hat, _ = self(x)

        loss = self.mse_loss(y_hat, y) + 0.1 * self.l1_loss(y_hat, y)
        mae = torch.mean(torch.abs(y_hat - y))

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_mse", self.mse_loss(y_hat, y), prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_mae", mae, on_step=False, on_epoch=True)
        self.log(f"{stage}_r2", self.r2(y_hat, y), on_step=False, on_epoch=True)
        self.log(f"{stage}_mape", self.mape(y_hat, y), on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=2,factor=0.5,verbose=True)
        return {"optimizer": optimizer,"lr_scheduler": {"scheduler": scheduler,"monitor": "val_loss", "interval": "epoch", "frequency": 1 }}