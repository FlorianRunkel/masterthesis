import torch
import torch.nn as nn
import pytorch_lightning as pl

class TFTModel(pl.LightningModule):
    def __init__(self, 
                 sequence_features=13,    # 5 Position + 8 Transition Features
                 global_features=9,
                 hidden_size=128,
                 num_layers=2,
                 dropout=0.2,
                 bidirectional=True,
                 lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.lr = lr
        self.loss_fn = nn.BCELoss()
        
        # Sequenz-Verarbeitung
        self.sequence_encoder = nn.LSTM(
            input_size=sequence_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Attention für Sequenz
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2 if bidirectional else hidden_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_size * 2 if bidirectional else hidden_size)
        
        # Globale Feature-Verarbeitung
        self.global_encoder = nn.Sequential(
            nn.Linear(global_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Feature Fusion
        fusion_input_size = (hidden_size * 2 if bidirectional else hidden_size) + hidden_size
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Karriere-Entwicklungs-Prädiktor
        self.career_predictor = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
        
        # Initialisiere Gewichte
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)

    def forward(self, batch):
        x_seq, x_global, lengths = batch
        batch_size = x_seq.size(0)
        
        # Pack die Sequenz für effizientes LSTM-Processing
        packed_seq = nn.utils.rnn.pack_padded_sequence(
            x_seq, 
            lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # Verarbeite Sequenz durch LSTM
        lstm_out, _ = self.sequence_encoder(packed_seq)
        
        # Unpack die Sequenz wieder
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        # Erstelle Attention Mask für gepaddte Bereiche
        max_len = lstm_out.size(1)
        attention_mask = torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]
        
        # Self-Attention auf LSTM output mit Maske
        attn_out, attn_weights = self.attention(
            lstm_out, 
            lstm_out, 
            lstm_out,
            key_padding_mask=~attention_mask  # Invertiere für PyTorch Attention
        )
        
        # Residual connection und Layer Norm
        sequence_features = self.layer_norm(lstm_out + attn_out)  # [batch, seq_len, hidden*2]
        
        # Gewichteter Durchschnitt der Sequenz-Features basierend auf der Attention
        sequence_features = torch.sum(sequence_features * attn_weights.mean(dim=1).unsqueeze(-1), dim=1)
        
        # Verarbeite globale Features
        global_features = self.global_encoder(x_global)  # [batch, hidden]
        
        # Kombiniere Features
        combined = torch.cat([sequence_features, global_features], dim=1)
        fused = self.fusion_layer(combined)
        
        # Vorhersage
        return self.career_predictor(fused)

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")
        
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")
    
    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")
    
    def _shared_step(self, batch, stage):
        (x_seq, x_global, lengths), y = batch
        
        # Forward pass
        y_hat = self((x_seq, x_global, lengths))
        
        # Reshape wenn nötig
        if y.dim() == 1:
            y = y.unsqueeze(1)
            
        # Berechne Loss
        loss = self.loss_fn(y_hat, y)
        
        # Berechne Metriken
        preds = (y_hat > 0.5).float()
        acc = (preds == y).float().mean()
        
        # Logging
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        
        # Detailliertes Logging während des Trainings
        if stage == "train":
            self.log("learning_rate", self.optimizers().param_groups[0]['lr'])
            
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }