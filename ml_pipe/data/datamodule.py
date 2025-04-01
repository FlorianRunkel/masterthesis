import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch
from typing import Optional, Tuple

class LinkedInDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class LinkedInDataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_processor,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 train_split: float = 0.8,
                 val_split: float = 0.1):
        super().__init__()
        self.data_processor = data_processor
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_split = train_split
        self.val_split = val_split
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def setup(self, stage: Optional[str] = None):
        """Lädt und bereitet die Daten vor"""
        # Hier müssen Sie Ihre Daten laden und verarbeiten
        # Beispiel:
        # raw_data = pd.read_csv("path/to/your/data.csv")
        # processed_data = self.data_processor.process_raw_data(raw_data)
        # features = self.data_processor.prepare_features(...)
        
        # Dummy-Daten für das Beispiel
        n_samples = 1000
        n_features = 10
        features = np.random.randn(n_samples, n_features)
        targets = np.random.randn(n_samples, 1)
        
        # Split der Daten
        train_size = int(n_samples * self.train_split)
        val_size = int(n_samples * self.val_split)
        
        # Train/Val/Test Split
        indices = np.random.permutation(n_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Erstellen der Datasets
        self.train_dataset = LinkedInDataset(
            features[train_indices], 
            targets[train_indices]
        )
        self.val_dataset = LinkedInDataset(
            features[val_indices], 
            targets[val_indices]
        )
        self.test_dataset = LinkedInDataset(
            features[test_indices], 
            targets[test_indices]
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        ) 