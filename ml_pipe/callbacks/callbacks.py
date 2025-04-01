import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from typing import Dict, Any

def get_callbacks(config: Dict[str, Any]) -> list:
    """Erstellt und gibt eine Liste von Callbacks zurück"""
    callbacks = []
    
    # Model Checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=config['checkpoint_dir'],
        filename='linkedin-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
        save_last=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early Stopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=config['early_stopping_patience'],
        verbose=True,
        mode='min'
    )
    callbacks.append(early_stop_callback)
    
    return callbacks 