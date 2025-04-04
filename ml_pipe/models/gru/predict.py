import torch
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
import logging
import os
import sys
from pathlib import Path

# Füge das Hauptverzeichnis zum Python-Pfad hinzu
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.gru.model import GRUModel

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def predict_career_change_from_data(career_data, model_path=None, seq_len=5):
    """
    Vorhersage, ob ein Profil einen Karrierewechsel machen wird, basierend auf direkt eingegebenen Daten
    
    Args:
        career_data (list): Liste von Dictionaries mit Karriereschritten
            Jedes Dictionary sollte folgende Schlüssel enthalten:
            - position: Position/Titel
            - company: Unternehmen
            - start_date: Startdatum (Format: YYYY-MM-DD)
            - end_date: Enddatum (Format: YYYY-MM-DD) oder None für aktuelle Position
        model_path (str): Pfad zum trainierten Modell
        seq_len (int): Länge der Sequenz für die Vorhersage
        
    Returns:
        dict: Vorhersage und Konfidenz
    """
    # Setze Standardpfad für das Modell, falls keiner angegeben wurde
    if model_path is None:
        model_path = os.path.join(Path(__file__).parent, "models/career_prediction_model.ckpt")
    
    # Prüfe, ob Modell existiert
    if not os.path.exists(model_path):
        logger.error(f"Modell nicht gefunden: {model_path}")
        return {"error": "Modell nicht gefunden"}
    
    # Lade Modell
    logger.info(f"Lade Modell von {model_path}")
    model = GRUModel.load_from_checkpoint(model_path)
    model.eval()
    
    # Konvertiere Eingabedaten zu DataFrame
    df = pd.DataFrame(career_data)
    
    # Konvertiere Datumsspalten
    df['start_date'] = pd.to_datetime(df['start_date'])
    df['end_date'] = pd.to_datetime(df['end_date']).fillna(datetime.today())
    
    # Berechne Dauer in Monaten
    df['duration_months'] = (df['end_date'] - df['start_date']).dt.days // 30
    
    # Kodiere Positionen und Unternehmen
    df['position_level'] = df['position'].astype('category').cat.codes
    df['company_encoded'] = df['company'].astype('category').cat.codes
    
    # Feature-Engineering
    feature_cols = ['duration_months', 'position_level', 'company_encoded']
    
    # Extrahiere Features
    features = df[feature_cols].values
    
    # Erstelle Sequenz der Länge seq_len
    if len(features) < seq_len:
        # Padding für zu kurze Sequenzen
        pad = np.zeros((seq_len - len(features), len(feature_cols)))
        features = np.vstack((pad, features))
    else:
        # Nimm die letzten seq_len Einträge
        features = features[-seq_len:]
    
    # Konvertiere zu Tensor
    features = torch.FloatTensor(features).unsqueeze(0)  # Füge Batch-Dimension hinzu
    
    # Vorhersage
    with torch.no_grad():
        prediction = model(features)
        confidence = prediction.item()
    
    # Erstelle Ergebnis
    result = {
        "prediction": "Karrierewechsel wahrscheinlich" if confidence > 0.5 else "Kein Karrierewechsel wahrscheinlich",
        "confidence": confidence,
        "current_position": df.iloc[-1]['position'],
        "current_company": df.iloc[-1]['company'],
        "experience_months": df['duration_months'].sum()
    }
    
    logger.info(f"Vorhersage: {result['prediction']} (Konfidenz: {confidence:.2%})")
    
    return result

def predict_career_change(profile_id, db_path=None, model_path=None, seq_len=5):
    """
    Vorhersage, ob ein Profil einen Karrierewechsel machen wird
    
    Args:
        profile_id (int): ID des Profils
        db_path (str): Pfad zur SQLite-Datenbank
        model_path (str): Pfad zum trainierten Modell
        seq_len (int): Länge der Sequenz für die Vorhersage
        
    Returns:
        dict: Vorhersage und Konfidenz
    """
    # Setze Standardpfade, falls keine angegeben wurden
    if db_path is None:
        db_path = os.path.join(Path(__file__).parent.parent.parent, "data/database/career_data.db")
    if model_path is None:
        model_path = os.path.join(Path(__file__).parent, "models/career_prediction_model.ckpt")
    
    # Prüfe, ob Modell existiert
    if not os.path.exists(model_path):
        logger.error(f"Modell nicht gefunden: {model_path}")
        return {"error": "Modell nicht gefunden"}
    
    # Lade Modell
    logger.info(f"Lade Modell von {model_path}")
    model = GRUModel.load_from_checkpoint(model_path)
    model.eval()
    
    # Lade Daten aus der Datenbank
    logger.info(f"Lade Daten für Profil {profile_id} aus {db_path}")
    conn = sqlite3.connect(db_path)
    
    # Lade Karrierverlauf
    career_data = pd.read_sql_query(
        f"SELECT * FROM career_history WHERE profile_id = {profile_id} ORDER BY start_date", 
        conn
    )
    
    if len(career_data) == 0:
        logger.error(f"Keine Daten für Profil {profile_id} gefunden")
        return {"error": "Keine Daten gefunden"}
    
    # Konvertiere Datumsspalten
    career_data['start_date'] = pd.to_datetime(career_data['start_date'])
    career_data['end_date'] = pd.to_datetime(career_data['end_date']).fillna(datetime.today())
    
    # Berechne Dauer in Monaten
    career_data['duration_months'] = (career_data['end_date'] - career_data['start_date']).dt.days // 30
    
    # Kodiere Positionen und Unternehmen
    career_data['position_level'] = career_data['position'].astype('category').cat.codes
    career_data['company_encoded'] = career_data['company'].astype('category').cat.codes
    
    # Feature-Engineering
    feature_cols = ['duration_months', 'position_level', 'company_encoded']
    
    # Extrahiere Features
    features = career_data[feature_cols].values
    
    # Erstelle Sequenz der Länge seq_len
    if len(features) < seq_len:
        # Padding für zu kurze Sequenzen
        pad = np.zeros((seq_len - len(features), len(feature_cols)))
        features = np.vstack((pad, features))
    else:
        # Nimm die letzten seq_len Einträge
        features = features[-seq_len:]
    
    # Konvertiere zu Tensor
    features = torch.FloatTensor(features).unsqueeze(0)  # Füge Batch-Dimension hinzu
    
    # Vorhersage
    with torch.no_grad():
        prediction = model(features)
        confidence = prediction.item()
    
    # Schließe Datenbankverbindung
    conn.close()
    
    # Erstelle Ergebnis
    result = {
        "profile_id": profile_id,
        "prediction": "Karrierewechsel wahrscheinlich" if confidence > 0.5 else "Kein Karrierewechsel wahrscheinlich",
        "confidence": confidence,
        "current_position": career_data.iloc[-1]['position'],
        "current_company": career_data.iloc[-1]['company'],
        "experience_months": career_data['duration_months'].sum()
    }
    
    logger.info(f"Vorhersage für Profil {profile_id}: {result['prediction']} (Konfidenz: {confidence:.2%})")
    
    return result

if __name__ == "__main__":
    # Beispiel: Vorhersage für ein einzelnes Profil aus der Datenbank
    # result = predict_career_change(1)
    # print(result)
    
    # Beispiel: Vorhersage für ein Profil mit direkt eingegebenen Daten
    sample_career_data = [
        {
            "position": "Software Engineer",
            "company": "Tech Corp",
            "start_date": "2020-01-01",
            "end_date": "2021-06-30"
        },
        {
            "position": "Senior Software Engineer",
            "company": "Tech Corp",
            "start_date": "2021-07-01",
            "end_date": None  # Aktuelle Position
        }
    ]
    
    result = predict_career_change_from_data(sample_career_data)
    print(result) 