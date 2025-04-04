import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class CareerDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class DataModule(pl.LightningDataModule):
    def __init__(self, db_path='ml_pipe/data/database/career_data.db', batch_size=32, seq_len=5):
        super().__init__()
        self.db_path = db_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        
    def prepare_data(self):
        """
        Lädt die Daten aus der SQLite-Datenbank
        """
        logger.info(f"Lade Daten aus {self.db_path}")
        self.conn = sqlite3.connect(self.db_path)
        
        # Lade Karrierverläufe
        self.career_data = pd.read_sql_query(
            "SELECT * FROM career_history ORDER BY profile_id, start_date", 
            self.conn
        )
        
        # Konvertiere Datumsspalten
        self.career_data['start_date'] = pd.to_datetime(self.career_data['start_date'])
        self.career_data['end_date'] = pd.to_datetime(self.career_data['end_date']).fillna(datetime.today())
        
        # Berechne Dauer in Monaten
        self.career_data['duration_months'] = (self.career_data['end_date'] - self.career_data['start_date']).dt.days // 30
        
        # Kodiere Positionen und Unternehmen
        self.career_data['position_level'] = self.career_data['position'].astype('category').cat.codes
        self.career_data['company_encoded'] = self.career_data['company'].astype('category').cat.codes
        
        logger.info(f"Geladen: {len(self.career_data)} Karriereinträge von {self.career_data['profile_id'].nunique()} Profilen")
        
    def setup(self, stage=None):
        """
        Bereitet die Daten für das Training vor
        """
        if not hasattr(self, 'career_data'):
            self.prepare_data()
        
        # Feature-Engineering
        feature_cols = ['duration_months', 'position_level', 'company_encoded']
        
        # Erstelle Sequenzen für jedes Profil
        sequences = []
        targets = []
        
        for profile_id, group in self.career_data.groupby('profile_id'):
            # Sortiere nach Startdatum
            group = group.sort_values('start_date')
            
            # Extrahiere Features
            features = group[feature_cols].values
            
            # Erstelle Sequenz der Länge seq_len
            if len(features) < self.seq_len:
                # Padding für zu kurze Sequenzen
                pad = np.zeros((self.seq_len - len(features), len(feature_cols)))
                features = np.vstack((pad, features))
            else:
                # Nimm die letzten seq_len Einträge
                features = features[-self.seq_len:]
            
            sequences.append(features.astype(np.float32))
            
            # Zielvariable: Hat diese Person am Ende gewechselt?
            # Wir betrachten einen Wechsel als Ziel, wenn es mehr als einen Eintrag gibt
            target = 1.0 if len(group) > 1 else 0.0
            targets.append([target])
        
        # Konvertiere zu NumPy-Arrays
        self.sequences = np.array(sequences)
        self.targets = np.array(targets)
        
        # Teile in Train/Val/Test
        n_samples = len(self.sequences)
        train_size = int(0.7 * n_samples)
        val_size = int(0.15 * n_samples)
        
        # Erstelle Datasets
        self.train_dataset = CareerDataset(
            self.sequences[:train_size], 
            self.targets[:train_size]
        )
        
        self.val_dataset = CareerDataset(
            self.sequences[train_size:train_size+val_size], 
            self.targets[train_size:train_size+val_size]
        )
        
        self.test_dataset = CareerDataset(
            self.sequences[train_size+val_size:], 
            self.targets[train_size+val_size:]
        )
        
        logger.info(f"Daten aufgeteilt: {len(self.train_dataset)} Train, {len(self.val_dataset)} Val, {len(self.test_dataset)} Test")
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
    
    def teardown(self, stage=None):
        """
        Schließt die Datenbankverbindung
        """
        if hasattr(self, 'conn'):
            self.conn.close()
            logger.info("Datenbankverbindung geschlossen")