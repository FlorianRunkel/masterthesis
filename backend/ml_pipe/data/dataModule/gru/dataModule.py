import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pytorch_lightning import LightningDataModule
import logging
from backend.ml_pipe.data.featureEngineering.gru.featureEngineering_gru import FeatureEngineering

class CareerDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Die Daten sind bereits als [seq_len, 7] Feature-Tensoren vorbereitet
        features, label = self.data[idx]
        return features, label

class DataModule(LightningDataModule):
    def __init__(self, mongo_client, batch_size=32, train_split=0.7, val_split=0.15):
        super().__init__()
        self.mongo_client = mongo_client
        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    def setup(self, stage=None):
        if self.train_data is None:
            # Hole Rohdaten aus MongoDB
            result = self.mongo_client.get_all('time_dataset')
            raw_data = result.get('data', [])
            
            if not raw_data:
                print("Warnung: Keine Daten in der MongoDB Collection 'training_data2' gefunden – Setup wird übersprungen.")
            
            # Splitte die Daten
            n = len(raw_data)
            train_size = int(n * self.train_split)
            val_size = int(n * self.val_split)
            
            fe = FeatureEngineering()
            # Feature Engineering anwenden!
            train_features, train_labels = fe.extract_features_and_labels_for_training(raw_data[:train_size])
            val_features, val_labels = fe.extract_features_and_labels_for_training(raw_data[train_size:train_size + val_size])
            test_features, test_labels = fe.extract_features_and_labels_for_training(raw_data[train_size + val_size:])
            
            # Liste von Tupeln (features, label) erzeugen
            self.train_data = list(zip(train_features, train_labels))
            self.val_data = list(zip(val_features, val_labels))
            self.test_data = list(zip(test_features, test_labels))
            
            print(f"\nDatensatz aufgeteilt in:")
            print(f"- Training: {len(self.train_data)} Einträge")
            print(f"- Validierung: {len(self.val_data)} Einträge")
            print(f"- Test: {len(self.test_data)} Einträge")
    
    def train_dataloader(self):
        return DataLoader(
            CareerDataset(self.train_data),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )
    
    def val_dataloader(self):
        return DataLoader(
            CareerDataset(self.val_data),
            batch_size=self.batch_size,
            num_workers=4,
        )
    
    def test_dataloader(self):
        return DataLoader(
            CareerDataset(self.test_data),
            batch_size=self.batch_size,
            num_workers=4,
        )