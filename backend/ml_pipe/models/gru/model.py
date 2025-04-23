import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

class GRUModel(pl.LightningModule):
    def __init__(self, input_size=3, hidden_size=50, num_layers=4, dropout=0.2, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        
        # GRU Schichten
        self.gru1 = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        self.dropout1 = nn.Dropout(dropout)
        
        self.gru2 = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        self.dropout2 = nn.Dropout(dropout)
        
        self.gru3 = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        self.dropout3 = nn.Dropout(dropout)
        
        self.gru4 = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        self.dropout4 = nn.Dropout(dropout)
        
        # Output Layer
        self.fc = nn.Linear(hidden_size, 1)
        self.tanh = nn.Tanh()
        
        self.loss_fn = nn.MSELoss()  # Mean Squared Error Loss
        self.lr = lr

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if x.size(-1) != self.hparams.input_size:
            if x.size(-1) < self.hparams.input_size:
                pad = torch.zeros(x.size(0), x.size(1), self.hparams.input_size - x.size(-1), device=x.device)
                x = torch.cat([x, pad], dim=-1)
            else:
                x = x[:, :, :self.hparams.input_size]
        
        # Erste GRU Schicht
        out, _ = self.gru1(x)
        out = self.tanh(out)
        out = self.dropout1(out)
        
        # Zweite GRU Schicht
        out, _ = self.gru2(out)
        out = self.tanh(out)
        out = self.dropout2(out)
        
        # Dritte GRU Schicht
        out, _ = self.gru3(out)
        out = self.tanh(out)
        out = self.dropout3(out)
        
        # Vierte GRU Schicht
        out, _ = self.gru4(out)
        out = self.tanh(out)
        out = self.dropout4(out)
        
        # Nur die letzte Ausgabe der Sequenz verwenden
        out = out[:, -1, :]
        
        # Output Layer
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