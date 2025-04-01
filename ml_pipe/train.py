import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, '/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis')

from ml_pipe.data.dummy_data import create_dummy_database
import sqlite3
import logging
from datetime import datetime

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LinkedInDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class LinkedInModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims)-1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def prepare_data():
    """Bereitet die Daten aus der SQLite-Datenbank vor"""
    conn = sqlite3.connect('ml_pipe/data/database/linkedin_data.db')
    
    # Lade Profile und Erfahrungen
    profiles = pd.read_sql_query('SELECT * FROM profiles', conn)
    experiences = pd.read_sql_query('SELECT * FROM experiences', conn)
    
    # Bereite Features vor
    features = []
    labels = []
    
    for _, profile in profiles.iterrows():
        # Numerische Features
        profile_features = [
            float(profile['experience_years']),
            float(profile['connections_count']),
            float((datetime.now() - pd.to_datetime(profile['created_at'])).days)
        ]
        
        # Kategorische Features (One-Hot Encoding)
        position_level = profile['current_position']
        position_encoding = {
            'Entry Level': 0,
            'Mid Level': 1,
            'Senior Level': 2,
            'Lead': 3,
            'Manager': 4,
            'Director': 5,
            'VP': 6,
            'C-Level': 7
        }
        profile_features.append(float(position_encoding.get(position_level, 0)))
        profile_features.append(float(exp_count))
        
        # Berechne die Anzahl der Erfahrungen
        exp_count = len(experiences[experiences['profile_id'] == profile['id']])
        profile_features.append(exp_count)
        
        features.append(profile_features)
        
        # Label: Nächster Karriereschritt (vereinfacht)
        current_level = position_encoding.get(position_level, 0)
        next_level = min(current_level + 1, 7)  # Maximal C-Level
        labels.append(next_level)
    
    conn.close()
    
    return np.array(features), np.array(labels)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """Trainiert das Modell"""
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validierung
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        logger.info(f'Epoch [{epoch+1}/{num_epochs}], '
                   f'Train Loss: {train_loss/len(train_loader):.4f}, '
                   f'Val Loss: {val_loss:.4f}, '
                   f'Accuracy: {accuracy:.2f}%')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'ml_pipe/models/best_model.pth')
            logger.info('Bestes Modell gespeichert!')

def evaluate_model(model, test_loader, device):
    """Evaluiert das Modell auf dem Testset"""
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    logger.info(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy, all_predictions, all_labels

def main():
    
    # Bereite Daten vor
    features, labels = prepare_data()
    print(features, labels)
    
    # Teile Daten in Train/Val/Test
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    
    # Skaliere Features
    scaler = StandardScaler()
    print(X_train)

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Erstelle Datasets und DataLoaders
    train_dataset = LinkedInDataset(X_train, y_train)
    test_dataset = LinkedInDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Modell-Parameter
    input_dim = X_train.shape[1]
    hidden_dims = [64, 32, 16]
    output_dim = len(np.unique(labels))
    dropout_rate = 0.2
    
    # Initialisiere Modell
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LinkedInModel(input_dim, hidden_dims, output_dim, dropout_rate).to(device)
    
    # Loss und Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Trainiere Modell
    num_epochs = 5
    train_model(model, train_loader, criterion, optimizer, num_epochs, device)
    
    # Lade bestes Modell und evaluiere
    model.load_state_dict(torch.load('ml_pipe/models/best_model.pth'))
    accuracy, predictions, true_labels = evaluate_model(model, test_loader, device)
    
    # Speichere Modell und Scaler
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'input_dim': input_dim,
        'hidden_dims': hidden_dims,
        'output_dim': output_dim
    }, 'ml_pipe/models/model_checkpoint.pth')
    
    logger.info('Training und Evaluation abgeschlossen!')

if __name__ == '__main__':
    main() 