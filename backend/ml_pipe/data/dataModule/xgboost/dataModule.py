import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pytorch_lightning import LightningDataModule
import logging

'''
Dataset for XGBoost
'''
class Dataset(Dataset):

    '''
    Initialize Dataset
    '''
    def __init__(self, data):
        self.data = data
        self.logger = logging.getLogger(__name__)

    '''
    Get length of dataset
    '''
    def __len__(self):
        return len(self.data)

    '''
    Safe get value from dictionary
    '''
    def _safe_get(self, dict_obj, key, prefix=""):
        if key not in dict_obj:
            msg = f"{prefix}Feature '{key}' fehlt im Datensatz!"
            self.logger.error(msg)
            raise KeyError(msg)
        return float(dict_obj[key])

    '''
    Get item from dataset
    '''
    def __getitem__(self, idx):
        item = self.data[idx]
        features = item.get('features', {})

        career_sequence = features.get('career_sequence', [])

        if not career_sequence:
            self.logger.error(f"No career sequence for index {idx}")
            raise ValueError(f"No career sequence for index {idx}")

        position_features = torch.tensor([
            [
                self._safe_get(seq, 'level', f"Seq {i}: "),
                self._safe_get(seq, 'branche', f"Seq {i}: "),
                self._safe_get(seq, 'duration_months', f"Seq {i}: "),
            ] for i, seq in enumerate(career_sequence)], dtype=torch.float32)

        global_features = torch.tensor([
            self._safe_get(features, 'total_positions'),
            self._safe_get(features, 'total_experience_years'),
            self._safe_get(features, 'highest_degree'),
            self._safe_get(features, 'age_category')
        ], dtype=torch.float32)

        y = torch.tensor(float(item.get('label', 0)), dtype=torch.float32)

        return (position_features, global_features), y

'''
Custom collate function for variable sequence lengths
'''
def collate_fn(batch):
    sequences, globals = [], []
    labels = []
    lengths = []

    for (seq, glob), label in batch:
        sequences.append(seq)
        globals.append(glob)
        labels.append(label)
        lengths.append(len(seq))  # save length of each sequence

    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)

    globals = torch.stack(globals)
    labels = torch.stack(labels)
    lengths = torch.tensor(lengths)  # convert lengths to tensor

    return (sequences_padded, globals, lengths), labels

'''
DataModule for XGBoost
'''
class DataModule(LightningDataModule):

    '''
    Initialize DataModule
    '''
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

    '''
    Only used for compatibility with LightningDataModule
    '''
    def prepare_data(self):
        pass

    '''
    Setup data for training, validation and testing on GPUs
    '''
    def setup(self, stage=None):
        if self.train_data is None:
            result = self.mongo_client.get_all('training_data2')
            all_data = result.get('data', [])

            if not all_data:
                raise ValueError("No data found in MongoDB Collection 'training_data'")

            np.random.shuffle(all_data) # random shuffle

            n = len(all_data)
            train_size = int(n * self.train_split)
            val_size = int(n * self.val_split)

            self.train_data = all_data[:train_size]
            self.val_data = all_data[train_size:train_size + val_size]
            self.test_data = all_data[train_size + val_size:]

            print(f"\nDataset split:")
            print(f"- Train:      {len(self.train_data)}")
            print(f"- Validation: {len(self.val_data)}")
            print(f"- Test:       {len(self.test_data)}")

            print(f"\nFeature dimensions:")
            print(f"- Sequence features per time step: {self.sequence_dim}")
            print(f"- Global features: {self.global_dim}")

    '''
    Train dataloader
    '''
    def train_dataloader(self):
        return DataLoader(
            Dataset(self.train_data),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn
        )

    '''
    Validation dataloader
    '''
    def val_dataloader(self):
        return DataLoader(
            Dataset(self.val_data),
            batch_size=self.batch_size,
            num_workers=4,
            collate_fn=collate_fn
        )

    '''
    Test dataloader
    '''
    def test_dataloader(self):
        return DataLoader(
            Dataset(self.test_data),
            batch_size=self.batch_size,
            num_workers=4,
            collate_fn=collate_fn
        )