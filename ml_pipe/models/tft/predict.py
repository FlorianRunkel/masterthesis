import torch
import os
from ml_pipe.data.featureEngineering.featureEngineering import featureEngineering
from ml_pipe.models.tft.model import TFTModel
from datetime import datetime
import traceback
import logging

def preprocess(user_data):
    fe = featureEngineering()
    features = fe.extract_features_from_single_user(user_data)
    
    if features is None:
        raise ValueError("Nicht genug Daten für Vorhersage")

    # Die Features sind bereits in der richtigen Form [normalized_duration, level, branche_code]
    # Keine weitere Selektion nötig, da wir nur diese 3 Features verwenden
    return features.tolist()

def predict(input_data, model_path="ml_pipe/models/tft/saved_models/tft_model_20250415_155249.pt"):
    try:
        # Initialisiere Feature Engineering
        fe = featureEngineering()
        
        # Extrahiere Features
        features = fe.extract_features_from_single_user(input_data)
        if features is None:
            return {"confidence": [0.0], "recommendations": ["Keine gültigen Karrieredaten vorhanden"]}
            
        # Modell initialisieren
        model = TFTModel(input_size=3, hidden_size=32)
        
        # Modell laden
        checkpoint = torch.load(model_path, weights_only=True)
        model.load_state_dict(checkpoint)
        model.eval()
        
        # Input vorbereiten - numpy Array zu Tensor konvertieren
        input_tensor = torch.from_numpy(features).float()
        
        # Vorhersage machen
        with torch.no_grad():
            pred = model(input_tensor)
        
        # Vorhersage interpretieren
        pred_value = float(pred.item())
        
        # Interpretation der Vorhersage und Empfehlungen
        if pred_value > 0.7:
            status = "sehr wahrscheinlich wechselbereit"
            recommendations = [
                "Der Kandidat zeigt starke Anzeichen für einen bevorstehenden Wechsel.",
                "Aktive Ansprache empfohlen."
            ]
        elif pred_value > 0.5:
            status = "wahrscheinlich wechselbereit"
            recommendations = [
                "Der Kandidat könnte für einen Wechsel offen sein.",
                "Regelmäßige Kontaktaufnahme empfohlen."
            ]
        elif pred_value > 0.3:
            status = "möglicherweise wechselbereit"
            recommendations = [
                "Der Kandidat zeigt keine klaren Anzeichen für einen Wechsel.",
                "Beobachtung der Situation empfohlen."
            ]
        else:
            status = "bleibt wahrscheinlich"
            recommendations = [
                "Der Kandidat zeigt wenig Interesse an einem Wechsel.",
                "Längerfristige Beziehungspflege empfohlen."
            ]
        
        return {
            "confidence": [pred_value],
            "recommendations": recommendations
        }
        
    except Exception as e:
        print(f"Fehler bei der Vorhersage: {str(e)}")
        return {"confidence": [0.0], "recommendations": [f"Fehler bei der Vorhersage: {str(e)}"]}