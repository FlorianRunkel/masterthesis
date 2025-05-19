import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pytorch_lightning import LightningDataModule
import logging
from backend.ml_pipe.data.featureEngineering.gru.featureEngineering_gru import FeatureEngineering

class CareerDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.logger = logging.getLogger(__name__)
        self.fe = FeatureEngineering()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        features = item['features']
        label = item['label']

        career_sequence = features.get('career_sequence', [])
        if not career_sequence:
            self.logger.error(f"Keine Karriere-Sequenz für Index {idx}")
            return None

        seq_features = torch.tensor([
            [
                seq['level'],
                seq['branche'],
                seq['duration_months'],
                seq['time_since_start'],
                seq['time_until_end'],
                seq['is_current']
            ] for seq in career_sequence
        ], dtype=torch.float32)

        static_features = torch.tensor([
            features['total_positions'],
            features['company_changes'],
            features['total_experience_years'],
            features['highest_degree'],
            features['age_category'],
            features['avg_position_duration_months']
        ], dtype=torch.float32)

        y = torch.tensor(float(label), dtype=torch.float32)
        #print((seq_features, static_features), y)
        return (seq_features, static_features), y

def collate_fn(batch):
    sequences, statics = [], []
    labels = []
    
    for (seq, static), label in batch:
        sequences.append(seq)
        statics.append(static)
        labels.append(label)
    
    # Padding für variable Sequenzlängen
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    statics = torch.stack(statics)
    labels = torch.stack(labels)
    
    return (sequences_padded, statics), labels

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
            result = self.mongo_client.get_all('training_data2')
            raw_data = result.get('data', [])
            
            if not raw_data:
                print("Warnung: Keine Daten in der MongoDB Collection 'training_data2' gefunden – Setup wird übersprungen.")
            
            # Splitte die Daten
            n = len(raw_data)
            train_size = int(n * self.train_split)
            val_size = int(n * self.val_split)
            
            self.train_data = raw_data[:train_size]
            self.val_data = raw_data[train_size:train_size + val_size]
            self.test_data = raw_data[train_size + val_size:]
            
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
            collate_fn=collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            CareerDataset(self.val_data),
            batch_size=self.batch_size,
            num_workers=4,
            collate_fn=collate_fn
        )
    
    def test_dataloader(self):
        return DataLoader(
            CareerDataset(self.test_data),
            batch_size=self.batch_size,
            num_workers=4,
            collate_fn=collate_fn
        )