import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

class TFTModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size=32, dropout=0.1, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Feature Encoding
        self.input_projection = nn.Linear(input_size, hidden_size)

        # LSTM Layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        # Attention Layer
        self.attn = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)

        # Gating Mechanism (optional)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

        # Output Head
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

        self.loss_fn = nn.BCELoss()
        self.lr = lr

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        if x.size(-1) != self.hparams.input_size:
            diff = self.hparams.input_size - x.size(-1)
            if diff > 0:
                pad = torch.zeros(x.size(0), x.size(1), diff, device=x.device)
                x = torch.cat([x, pad], dim=-1)
            else:
                x = x[:, :, :self.hparams.input_size]

        x_proj = self.input_projection(x)  # Shape: [B, T, H]
        lstm_out, _ = self.lstm(x_proj)    # Shape: [B, T, H]

        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)  # Self-attention
        gated = self.gate(attn_out)
        fusion = gated * attn_out + (1 - gated) * lstm_out

        final_output = fusion[:, -1, :]  # Take last time step
        out = self.output_layer(final_output)  # [B, 1]
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