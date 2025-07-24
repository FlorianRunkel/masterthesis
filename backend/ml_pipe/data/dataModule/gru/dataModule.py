import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from pytorch_lightning import LightningDataModule
import logging
from backend.ml_pipe.data.featureEngineering.gru.featureEngineering_gru import FeatureEngineering
import joblib

'''
Helper
'''
def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.tensor(x, dtype=torch.float32)

'''
Dataset for GRU Model
'''
class Dataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, label = self.data[idx]
        label = np.log1p(label)
        if torch.isnan(to_tensor(features)).any() or torch.isinf(to_tensor(features)).any():
            logging.warning("Warning: NaN or Inf in Features!")
        if torch.isnan(to_tensor([label])).any() or torch.isinf(to_tensor([label])).any():
            logging.warning("Warning: NaN or Inf in Label!")
        return to_tensor(features), to_tensor([label])

'''
DataModule for GRU Model
'''
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

    '''
    Setup data for training, validation and testing
    '''
    def setup(self, stage=None):
        if self.train_data is None:

            result = self.mongo_client.get_all('timeseries_dataset')
            raw_data = result.get('data', [])

            if not raw_data:
                logging.warning("No data found in MongoDB Collection 'timeseries_dataset'")
                return

            n = len(raw_data)
            train_size = int(n * self.train_split)
            val_size = int(n * self.val_split)

            fe = FeatureEngineering()
            train_features, train_labels = fe.extract_sequences_by_profile_new(raw_data[:train_size])
            val_features, val_labels = fe.extract_sequences_by_profile_new(raw_data[train_size:train_size + val_size])
            test_features, test_labels = fe.extract_sequences_by_profile_new(raw_data[train_size + val_size:])

            self.scaler = MinMaxScaler()
            train_features_np = np.array(train_features).squeeze()  # (N, seq_len, features) → (N, F)
            val_features_np = np.array(val_features).squeeze()
            test_features_np = np.array(test_features).squeeze()

            self.scaler.fit(train_features_np)
            train_scaled = self.scaler.transform(train_features_np)
            val_scaled = self.scaler.transform(val_features_np)
            test_scaled = self.scaler.transform(test_features_np)

            train_scaled = [np.expand_dims(f, axis=0) for f in train_scaled]
            val_scaled = [np.expand_dims(f, axis=0) for f in val_scaled]
            test_scaled = [np.expand_dims(f, axis=0) for f in test_scaled]

            joblib.dump(self.scaler, "scaler_gru.joblib")

            self.train_data = list(zip(train_scaled, train_labels))
            self.val_data = list(zip(val_scaled, val_labels))
            self.test_data = list(zip(test_scaled, test_labels))

            logging.info(f"Dataset split:")
            logging.info(f"- Train:      {len(self.train_data)}")
            logging.info(f"- Validation: {len(self.val_data)}")
            logging.info(f"- Test:       {len(self.test_data)}")

            logging.debug(f"Example feature shape: {np.array(self.train_data[0][0]).shape}")
            logging.debug(f"Example label (before scaling): {self.train_data[0][1]}")

    '''
    Train dataloader
    '''
    def train_dataloader(self):
        return DataLoader(
            Dataset(self.train_data),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )

    '''
    Validation dataloader
    '''
    def val_dataloader(self):
        return DataLoader(
            Dataset(self.val_data),
            batch_size=self.batch_size,
            num_workers=4,
        )

    '''
    Test dataloader
    '''
    def test_dataloader(self):
        return DataLoader(
            Dataset(self.test_data),
            batch_size=self.batch_size,
            num_workers=4,
        )