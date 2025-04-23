import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

class TFTModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.loss_fn = nn.BCELoss()

        # Input Projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Stacked LSTM
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Self-Attention (more heads, deeper output)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_size)

        # Feedforward block after attention
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)

        # Output layer
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        # Padding/truncation
        if x.size(-1) != self.hparams.input_size:
            diff = self.hparams.input_size - x.size(-1)
            if diff > 0:
                pad = torch.zeros(x.size(0), x.size(1), diff, device=x.device)
                x = torch.cat([x, pad], dim=-1)
            else:
                x = x[:, :, :self.hparams.input_size]

        x = self.input_projection(x)
        lstm_out, _ = self.lstm(x)

        # Self-attention + residual
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        attn_out = self.attn_norm(attn_out + lstm_out)

        # Feedforward + residual
        ffn_out = self.ffn(attn_out)
        fusion = self.ffn_norm(ffn_out + attn_out)

        # Final time step
        final = fusion[:, -1, :]
        out = self.output_head(final)
        return out

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