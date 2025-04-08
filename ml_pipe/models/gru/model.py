import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

class GRUModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.BCELoss()
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
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

    def step(self, batch, stage):
        x, y = batch
        if isinstance(x, (list, tuple)):
            x = x[0]
        x = x.float()
        y = y.float().unsqueeze(1) if y.dim() == 1 else y
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        preds = (y_hat > 0.5).float()
        acc = (preds == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=x.size(0))
        self.log(f"{stage}_acc", acc, prog_bar=True, batch_size=x.size(0))
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def configure_optimizers(self):
        opt = optim.Adam(self.parameters(), lr=self.lr)
        sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, verbose=True)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "monitor": "val_loss"}}