import torch
import torch.nn as nn
import pytorch_lightning as pl

class GRUModel(pl.LightningModule):
    def __init__(self, seq_input_size=7, hidden_size=256, num_layers=2, dropout=0.2, lr=0.001):
        super().__init__()
        self.save_hyperparameters()

        self.gru = nn.GRU(
            input_size=seq_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # auf True setzen für BiGRU!
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

        self.loss_fn = nn.MSELoss()
        self.lr = lr

    def forward(self, x_seq):
        gru_out, _ = self.gru(x_seq)
        last_out = gru_out[:, -1, :]
        return self.fc(last_out)


    def step(self, batch, stage):
        x_seq, y = batch
        y_hat = self(x_seq.float()).squeeze(1)
        y = y.float().squeeze(1)
        loss = self.loss_fn(y_hat, y)
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=x_seq.size(0))
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)