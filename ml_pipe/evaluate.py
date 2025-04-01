import yaml
import pytorch_lightning as pl
import torch
import pandas as pd
import numpy as np
from typing import Dict, Any

from data.datamodule import LinkedInDataModule
from data.data_processor import LinkedInDataProcessor
from models.model import LinkedInModel

def load_model_from_checkpoint(checkpoint_path: str, config: Dict[str, Any]) -> LinkedInModel:
    """Lädt ein trainiertes Modell von einem Checkpoint"""
    model = LinkedInModel(
        input_dim=config['model']['input_dim'],
        hidden_dims=config['model']['hidden_dims'],
        output_dim=config['model']['output_dim'],
        dropout_rate=config['model']['dropout_rate'],
        learning_rate=config['model']['learning_rate']
    )
    
    model = model.load_from_checkpoint(checkpoint_path)
    return model

def evaluate_model(model: LinkedInModel, datamodule: LinkedInDataModule, trainer: pl.Trainer):
    """Führt die Evaluierung des Modells durch"""
    # Test durchführen
    test_results = trainer.test(model, datamodule)
    
    # Vorhersagen auf Test-Daten
    model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for batch in datamodule.test_dataloader():
            x, y = batch
            y_hat = model(x)
            test_predictions.extend(y_hat.numpy())
            test_targets.extend(y.numpy())
    
    return {
        'test_results': test_results,
        'predictions': np.array(test_predictions),
        'targets': np.array(test_targets)
    }

def main():
    # Konfiguration laden
    with open("src/config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Datenprozessor und DataModule initialisieren
    data_processor = LinkedInDataProcessor()
    datamodule = LinkedInDataModule(
        data_processor=data_processor,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split']
    )
    
    # Trainer initialisieren
    trainer = pl.Trainer(
        accelerator=config['training']['accelerator'],
        devices=config['training']['devices'],
        precision=config['training']['precision']
    )
    
    # Bestes Modell laden
    checkpoint_path = "checkpoints/linkedin-model-best.ckpt"  # Anpassen an Ihren besten Checkpoint
    model = load_model_from_checkpoint(checkpoint_path, config)
    
    # Evaluierung durchführen
    results = evaluate_model(model, datamodule, trainer)
    
    # Ergebnisse speichern
    np.save("evaluation_results/predictions.npy", results['predictions'])
    np.save("evaluation_results/targets.npy", results['targets'])
    
    # Metriken ausgeben
    print("\nEvaluierungsergebnisse:")
    print(f"Test Loss: {results['test_results'][0]['test_loss']:.4f}")

if __name__ == "__main__":
    main() 