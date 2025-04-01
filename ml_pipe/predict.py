import yaml
import torch
import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime

from ml_pipe.data.data_processor import LinkedInDataProcessor
from ml_pipe.models.model import LinkedInModel

def load_model_from_checkpoint(checkpoint_path: str, config: Dict[str, Any]) -> LinkedInModel:
    """Lädt ein trainiertes Modell von einem Checkpoint"""
    model = LinkedInModel(
        input_dim=config['model']['input_dim'],
        hidden_dims=config['model']['hidden_dims'],
        output_dim=config['model']['output_dim'],
        dropout_rate=config['model']['dropout_rate'],
        learning_rate=config['model']['learning_rate']
    )
    
    model = model.load_from_checkpoint(checkpoint_path)
    return model

def prepare_data_for_prediction(data: pd.DataFrame, data_processor: LinkedInDataProcessor, config: Dict[str, Any]) -> torch.Tensor:
    """Bereitet die Daten für die Vorhersage vor"""
    # Daten verarbeiten
    processed_data = data_processor.process_raw_data(data)
    
    # Features vorbereiten
    numerical_features, categorical_features = data_processor.prepare_features(
        processed_data,
        config['data']['categorical_columns'],
        config['data']['numerical_columns']
    )
    
    # Feature-Matrix erstellen
    features = data_processor.create_feature_matrix(numerical_features, categorical_features)
    
    # In Tensor umwandeln
    return torch.FloatTensor(features)

def predict(model: LinkedInModel, features: torch.Tensor) -> np.ndarray:
    """Führt Vorhersagen durch"""
    model.eval()
    with torch.no_grad():
        predictions = model(features)
    return predictions.numpy()

def predict_new_data(user_data):
    """
    Macht Vorhersagen für neue LinkedIn-Profile
    
    Args:
        user_data (pd.DataFrame): DataFrame mit den Nutzerdaten
        
    Returns:
        dict: Vorhersagen und Konfidenzwerte
    """
    # Lade das trainierte Modell und den Scaler
    checkpoint = torch.load('ml_pipe/models/model_checkpoint.pth')
    model_state_dict = checkpoint['model_state_dict']
    scaler = checkpoint['scaler']
    input_dim = checkpoint['input_dim']
    hidden_dims = checkpoint['hidden_dims']
    output_dim = checkpoint['output_dim']
    
    # Initialisiere das Modell
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LinkedInModel(input_dim, hidden_dims, output_dim).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # Bereite Features vor
    features = []
    for _, row in user_data.iterrows():
        profile_features = [
            row['experience_years'],
            row.get('connections_count', 0),
            (datetime.now() - pd.to_datetime(row['created_at'])).days
        ]
        
        # Position Level Encoding
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
        profile_features.append(position_encoding.get(row['current_position'], 0))
        
        # Anzahl der Erfahrungen
        profile_features.append(row.get('experience_count', 1))
        
        features.append(profile_features)
    
    # Skaliere Features
    features = scaler.transform(features)
    
    # Mache Vorhersagen
    with torch.no_grad():
        features_tensor = torch.FloatTensor(features).to(device)
        outputs = model(features_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)
        confidences = torch.max(probabilities, dim=1)[0]
    
    # Konvertiere Vorhersagen in lesbare Karriereschritte
    position_levels = [
        'Entry Level',
        'Mid Level',
        'Senior Level',
        'Lead',
        'Manager',
        'Director',
        'VP',
        'C-Level'
    ]
    
    next_career_steps = [position_levels[pred.item()] for pred in predictions]
    
    # Generiere Empfehlungen basierend auf den Vorhersagen
    recommendations = []
    for i, (current_pos, next_pos, conf) in enumerate(zip(user_data['current_position'], next_career_steps, confidences)):
        if current_pos != next_pos:
            recommendations.append(
                f"Basierend auf Ihrem Profil empfehlen wir den nächsten Karriereschritt: {next_pos}. "
                f"Konfidenz: {conf.item():.2%}"
            )
        else:
            recommendations.append(
                f"Sie sind bereits auf dem optimalen Karriereniveau ({current_pos}). "
                f"Konfidenz: {conf.item():.2%}"
            )
    
    return {
        'next_career_step': next_career_steps,
        'confidence': confidences.tolist(),
        'recommendations': recommendations
    }

def main():
    # Konfiguration laden
    with open("src/config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Datenprozessor initialisieren
    data_processor = LinkedInDataProcessor()
    
    # Bestes Modell laden
    checkpoint_path = "checkpoints/linkedin-model-best.ckpt"  # Anpassen an Ihren besten Checkpoint
    model = load_model_from_checkpoint(checkpoint_path, config)
    
    # Neue Daten laden
    new_data = pd.read_csv("data/predict/new_data.csv")  # Anpassen an Ihren Datenpfad
    
    # Daten für Vorhersage vorbereiten
    features = prepare_data_for_prediction(new_data, data_processor, config)
    
    # Vorhersagen durchführen
    predictions = predict(model, features)
    
    # Vorhersagen in ursprüngliches Format zurücktransformieren
    predictions = data_processor.inverse_transform_predictions(
        predictions,
        config['data']['target_column']
    )
    
    # Ergebnisse speichern
    results_df = pd.DataFrame({
        'id': new_data.index,
        'prediction': predictions.flatten()
    })
    results_df.to_csv("predictions/predictions.csv", index=False)
    
    print("\nVorhersagen wurden in 'predictions/predictions.csv' gespeichert.")

if __name__ == "__main__":
    main() 