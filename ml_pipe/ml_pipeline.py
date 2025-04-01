import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any
import pandas as pd
import numpy as np

class LinkedInDataset(Dataset):
    def __init__(self, data: pd.DataFrame, target_column: str):
        self.data = data
        self.target_column = target_column
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Hier müssen die Features entsprechend Ihrer Daten angepasst werden
        features = torch.tensor(self.data.iloc[idx].drop(self.target_column).values, dtype=torch.float32)
        target = torch.tensor(self.data.iloc[idx][self.target_column], dtype=torch.float32)
        return features, target

class LinkedInModel(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 1):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.layer3 = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.dropout(x, p=0.2)
        x = F.relu(self.layer2(x))
        x = F.dropout(x, p=0.2)
        x = self.layer3(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

class LinkedInMLPipeline:
    def __init__(self, data_path: str, target_column: str):
        self.data_path = data_path
        self.target_column = target_column
        self.model = None
        self.trainer = None
        
    def prepare_data(self) -> pd.DataFrame:
        """Laden und Vorbereiten der LinkedIn-Daten"""
        data = pd.read_csv(self.data_path)
        # Hier können Sie weitere Datenvorbereitungsschritte hinzufügen
        return data
    
    def setup_model(self, input_dim: int):
        """Initialisierung des PyTorch Lightning Modells"""
        self.model = LinkedInModel(input_dim=input_dim)
        self.trainer = pl.Trainer(
            max_epochs=100,
            accelerator='auto',
            devices=1,
            log_every_n_steps=10
        )
    
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame):
        """Training des Modells"""
        train_dataset = LinkedInDataset(train_data, self.target_column)
        val_dataset = LinkedInDataset(val_data, self.target_column)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        
        self.trainer.fit(self.model, train_loader, val_loader)
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Vorhersagen mit dem trainierten Modell"""
        self.model.eval()
        dataset = LinkedInDataset(data, self.target_column)
        loader = DataLoader(dataset, batch_size=32)
        
        predictions = []
        with torch.no_grad():
            for batch, _ in loader:
                pred = self.model(batch)
                predictions.extend(pred.numpy())
        
        return np.array(predictions) 