import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pytorch_lightning import LightningDataModule
import logging

class Dataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.logger = logging.getLogger(__name__)
        
    def __len__(self):
        return len(self.data)
    
    def _safe_get(self, dict_obj, key, prefix=""):
        """Sicheres Abrufen von Werten. Werfe Exception, wenn ein Wert fehlt."""
        if key not in dict_obj:
            msg = f"{prefix}Feature '{key}' fehlt im Datensatz!"
            self.logger.error(msg)
            raise KeyError(msg)
        return float(dict_obj[key])
    
    def __getitem__(self, idx):
        item = self.data[idx]
        features = item.get('features', {})
        
        # Extrahiere die Sequenz-Features
        career_sequence = features.get('career_sequence', [])
        
        if not career_sequence:
            self.logger.error(f"Keine Karriere-Sequenz für Index {idx}")
            raise ValueError(f"Keine Karriere-Sequenz für Index {idx}")
        
        # Erstelle Tensoren für die tatsächliche Sequenzlänge (nur Positionsfeatures)
        position_features = torch.tensor([
            [
                self._safe_get(seq, 'level', f"Seq {i}: "),
                self._safe_get(seq, 'branche', f"Seq {i}: "),
                self._safe_get(seq, 'duration_months', f"Seq {i}: "),
            ] for i, seq in enumerate(career_sequence)], dtype=torch.float32)
        
        # Globale Features (nur die erlaubten)
        global_features = torch.tensor([
            self._safe_get(features, 'total_positions'),
            self._safe_get(features, 'total_experience_years'),
            self._safe_get(features, 'highest_degree'),
            self._safe_get(features, 'age_category')
        ], dtype=torch.float32)
        
        # Label (Default 0 für "kein Wechsel")
        y = torch.tensor(float(item.get('label', 0)), dtype=torch.float32)
        
        return (position_features, global_features), y

def collate_fn(batch):
    """
    Custom collate Funktion für variable Sequenzlängen
    """
    # Entpacke die Batch-Daten
    sequences, globals = [], []
    labels = []
    lengths = []  # Liste für die Sequenzlängen
    
    for (seq, glob), label in batch:
        sequences.append(seq)
        globals.append(glob)
        labels.append(label)
        lengths.append(len(seq))  # Speichere die Länge jeder Sequenz
    
    # Packe die Sequenzen mit Padding
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    
    # Erstelle Tensoren
    globals = torch.stack(globals)
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths)  # Konvertiere Längen zu Tensor
    
    return (sequences_padded, globals, lengths), labels

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
        self.sequence_dim = 6  # 6 Position Features
        self.global_dim = 3     # Globale Features
        
    def prepare_data(self):
        # Diese Methode wird nur auf einem GPU ausgeführt
        # Hier können wir Daten herunterladen oder verarbeiten
        pass
        
    def setup(self, stage=None):
        # Diese Methode wird auf jedem GPU ausgeführt
        if self.train_data is None:
            # Hole alle Trainingsdaten aus MongoDB
            result = self.mongo_client.get_all('training_data2')
            all_data = result.get('data', [])
            
            if not all_data:
                raise ValueError("Keine Daten in der MongoDB Collection 'training_data' gefunden")
            
            # Mische die Daten
            np.random.shuffle(all_data)
            
            # Berechne Split-Indizes
            n = len(all_data)
            train_size = int(n * self.train_split)
            val_size = int(n * self.val_split)
            
            # Splitte die Daten
            self.train_data = all_data[:train_size]
            self.val_data = all_data[train_size:train_size + val_size]
            self.test_data = all_data[train_size + val_size:]
            
            print(f"\nDatensatz aufgeteilt in:")
            print(f"- Training: {len(self.train_data)} Einträge")
            print(f"- Validierung: {len(self.val_data)} Einträge")
            print(f"- Test: {len(self.test_data)} Einträge")
            print(f"\nFeature-Dimensionen:")
            print(f"- Sequenz-Features pro Zeitschritt: {self.sequence_dim}")
            print(f"- Globale Features: {self.global_dim}")
    
    def train_dataloader(self):
        return DataLoader(
            Dataset(self.train_data),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            Dataset(self.val_data),
            batch_size=self.batch_size,
            num_workers=4,
            collate_fn=collate_fn
        )
    
    def test_dataloader(self):
        return DataLoader(
            Dataset(self.test_data),
            batch_size=self.batch_size,
            num_workers=4,
            collate_fn=collate_fn
        )