import os
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import sys
from pathlib import Path

# Navigiere 2 Ebenen nach oben zu "ml_pipe"
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

# Eigene Module importieren
from data.dataCleaningHandler import DataCleaningHandler
from data.datamodule import DataModule
from models.gru.model import GRUModel


def run_pipeline():
    # 1. Datenbereinigung
    data_cleaner = DataCleaningHandler()
    data_cleaner.clean_all()
    data_cleaner.close()
    
    # 2. Feature Engineering und Datenvorbereitung
    data_module = DataModule(batch_size=32, seq_len=5)
    data_module.setup()

    # 3. Dynamisch die Input-Feature-Anzahl abrufen (wenn in DataModule verfügbar)
    input_size = getattr(data_module, "input_size", 3)

    # 4. Modell initialisieren
    model = GRUModel(
        input_size=input_size,  # z. B. 3: duration_months, position_level, company_encoded
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        lr=1e-3
    )

    # 5. Logging und Callbacks
    logger = TensorBoardLogger("logs", name="career_gru")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            mode="min"
        ),
        ModelCheckpoint(
            monitor="val_loss",
            dirpath=os.path.join(os.path.dirname(__file__), "checkpoints"),
            filename="career_prediction_gru",
            save_top_k=1,
            mode="min"
        )
    ]

    # 6. Trainer konfigurieren
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="cpu",  # oder "auto" für GPU-Nutzung
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10
    )

    # 7. Training starten
    trainer.fit(model, datamodule=data_module)

    # 8. Evaluation (falls test_dataloader vorhanden)
    trainer.test(model, datamodule=data_module)

    # 9. Modell speichern mit Zeitstempel
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join(os.path.dirname(__file__), f"models/run_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    trainer.save_checkpoint(os.path.join(model_dir, "career_prediction_gru.ckpt"))


if __name__ == "__main__":
    run_pipeline()