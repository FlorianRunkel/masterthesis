import torch
import numpy as np
import sys
sys.path.insert(0, '/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/')

from backend.ml_pipe.data.database.mongodb import MongoDb
import pytorch_lightning as pl
import json

class CareerDataset(torch.utils.data.Dataset):
    def __init__(self, samples, position_to_idx):
        self.samples = samples
        self.position_to_idx = position_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        pos_idx = self.position_to_idx[sample['aktuelle_position']]
        wechselzeitraum = sample['wechselzeitraum']
        x_seq = torch.tensor([pos_idx, wechselzeitraum], dtype=torch.float32)
        y = torch.tensor(sample['label'] - 1, dtype=torch.long)  # 0-3 statt 1-4
        return x_seq, y
    
import torch
import pytorch_lightning as pl
import numpy as np
import json

class CareerDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        # Daten aus MongoDB laden
        mongo_client = MongoDb(user='florianrunkel', password='ur04mathesis', db_name='Database')

        result = mongo_client.get_all('career_labels_tft')
        raw_data = result.get('data', [])
        len(raw_data)
        print(raw_data[0])

        # Aufteilen in Train/Val/Test (70/15/15)
        n = len(raw_data)
        train_size = int(0.7 * n)
        val_size = int(0.15 * n)

        np.random.shuffle(raw_data)
        self.train_data = raw_data[:train_size]
        self.val_data = raw_data[train_size:train_size + val_size]
        self.test_data = raw_data[train_size + val_size:]

        print(f"\nDatensatz aufgeteilt in:")
        print(f"- Training: {len(self.train_data)} Einträge")
        print(f"- Validierung: {len(self.val_data)} Einträge")
        print(f"- Test: {len(self.test_data)} Einträge")

        all_positions = set(s['aktuelle_position'] for s in raw_data)
        self.position_to_idx = {pos: idx for idx, pos in enumerate(sorted(all_positions))}

        # Datasets erstellen (Mapping übergeben!)
        self.train_dataset = CareerDataset(self.train_data, self.position_to_idx)
        self.val_dataset = CareerDataset(self.val_data, self.position_to_idx)
        self.test_dataset = CareerDataset(self.test_data, self.position_to_idx)

        # Nach self.position_to_idx = ...
        with open("backend/ml_pipe/data/dataModule/tft/position_to_idx.json", "w") as f:
            json.dump(self.position_to_idx, f)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4
        )