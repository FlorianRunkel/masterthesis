import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

class GRUModel(pl.LightningModule):
    def __init__(self, input_size=6, hidden_size=128, num_layers=3, dropout=0.3, lr=0.001):
        super().__init__()
        self.save_hyperparameters()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        
        self.fc = nn.Linear(hidden_size, 1)
        
        self.loss_fn = nn.BCEWithLogitsLoss()  # Für Binärklassifikation
        self.lr = lr

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # Nur letzter Zeitschritt
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def step(self, batch, stage):
        x, y = batch
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = x.float()
        y = y.float().unsqueeze(1) if y.dim() == 1 else y
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=x.size(0))
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def configure_optimizers(self):
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=0.9,
            nesterov=False,
            weight_decay=1e-7
        )
        return optimizer