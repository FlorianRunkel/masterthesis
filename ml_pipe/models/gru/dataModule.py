from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from featureEngineering import featureEngineering

class Dataset(Dataset):
    def __init__(self, mongo_client, collection_name="CareerData"):
        self.data = mongo_client.get({}, collection_name)
        self.sequences, self.labels = self._process_data(self.data)

    def _process_data(self, documents):
        fe = featureEngineering()
        X, y = fe.extract_features_and_labels(documents)
        return X,y

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class DataModule(pl.LightningDataModule):
    def __init__(self, mongo_client, batch_size=16):
        super().__init__()
        self.mongo_client = mongo_client
        self.batch_size = batch_size

    def setup(self, stage=None):
        full_data = Dataset(self.mongo_client)
        train_len = int(len(full_data) * 0.7)
        val_len = len(full_data) - train_len
        self.train_data, self.val_data = random_split(full_data, [train_len, val_len])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)