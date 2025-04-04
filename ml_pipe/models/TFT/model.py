import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from ml_pipe.models.gru.dataset_gru import Dataset 

# ------------------------------
# 2. GRU-Modell
# ------------------------------
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # Klassifikation

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # letztes Zeitschritt
        out = self.fc(out)
        return self.sigmoid(out)

# ------------------------------
# 3. Training
# ------------------------------
def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss/len(dataloader):.4f}")

# ------------------------------
# 4. Main Run
# ------------------------------
if __name__ == "__main__":
    input_size = 10
    seq_len = 5
    hidden_size = 64
    batch_size = 32

    dataset = Dataset(num_samples=1000, seq_len=seq_len, input_size=input_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GRUModel(input_size=input_size, hidden_size=hidden_size)
    criterion = nn.BCELoss()  # da Sigmoid → Binary Cross Entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, dataloader, criterion, optimizer, num_epochs=10)