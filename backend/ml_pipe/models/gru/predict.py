import torch
import os
import sys
import traceback
import numpy as np
from backend.ml_pipe.data.featureEngineering.featureEngineering import featureEngineering
from backend.ml_pipe.data.linkedInData.handler import extract_career_data, extract_education_data, extract_additional_features, estimate_age_category
from backend.ml_pipe.models.gru.model import GRUModel

# Füge den Backend-Ordner zum Python-Path hinzu
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(backend_dir)

# Definiere den Basispfad für die Modelle
MODEL_BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_models")

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        return self.model(x)

def calculate_feature_importance(model, seq_tensor):
    """Berechnet Feature-Wichtigkeiten basierend auf den Modelleingaben"""
    try:
        # Erstelle Feature-Namen und deren Beiträge basierend auf den tatsächlichen Daten
        sequence_features = seq_tensor[0].numpy()  # [sequence_length, features]
        
        # Berechne die durchschnittlichen Werte für Sequenz-Features
        avg_sequence_features = np.mean(np.abs(sequence_features), axis=0)
        
        # Normalisiere die Werte
        total_importance = np.sum(avg_sequence_features)
        if total_importance == 0:
            feature_importance = np.zeros_like(avg_sequence_features)
        else:
            feature_importance = (avg_sequence_features / total_importance) * 100
        
        # Feature-Namen definieren
        sequence_feature_names = [
            "Position Level",
            "Branche",
            "Beschäftigungsdauer",
            "Zeit seit Beginn",
            "Zeit bis Ende",
            "Aktuelle Position"
        ]
        
        # Erstelle Feature-Erklärungen
        explanations = []
        for name, importance in zip(sequence_feature_names, feature_importance):
            if importance > 1.0:  # Nur Features mit mehr als 1% Einfluss
                description = get_feature_description(name, importance)
                explanations.append({
                    "feature": name,
                    "impact_percentage": round(float(importance), 1),
                    "description": description
                })
        
        # Sortiere nach Wichtigkeit
        explanations.sort(key=lambda x: x["impact_percentage"], reverse=True)
        
        # Wenn keine Erklärungen gefunden wurden, füge eine Standard-Erklärung hinzu
        if not explanations:
            explanations.append({
                "feature": "Gesamtanalyse",
                "impact_percentage": 0.0,
                "description": "Die Vorhersage basiert auf einer Kombination verschiedener Karrierefaktoren."
            })
        
        return explanations
        
    except Exception as e:
        print(f"Fehler bei der Feature-Wichtigkeitsberechnung: {str(e)}")
        return [{
            "feature": "Karriereverlauf",
            "impact_percentage": 0.0,
            "description": "Die Vorhersage basiert auf der Analyse des gesamten Karriereverlaufs."
        }]

def get_feature_description(feature_name, importance):
    """Generiert Beschreibungen für Features basierend auf ihrer Wichtigkeit"""
    descriptions = {
        "Position Level": "Die Hierarchieebene der aktuellen und vorherigen Positionen hat einen signifikanten Einfluss auf die Wechselbereitschaft.",
        "Branche": "Die Branchenzugehörigkeit und deren Entwicklung sind wichtige Indikatoren für potenzielle Wechsel.",
        "Beschäftigungsdauer": "Die Verweildauer in Positionen gibt Aufschluss über das Wechselverhalten.",
        "Zeit seit Beginn": "Die Zeit seit Beginn der aktuellen Position beeinflusst die Wechselwahrscheinlichkeit.",
        "Zeit bis Ende": "Die Dauer bis zum Ende einer Position zeigt Muster im Wechselverhalten.",
        "Aktuelle Position": "Der aktuelle Karrierestatus ist ein wichtiger Indikator für Wechselbereitschaft."
    }
    return descriptions.get(feature_name, "Dieses Feature beeinflusst die Vorhersage.")

def predict(input_data, model_name="gru_model_20250502_115815.pt"):
    try:
        # Feature Engineering Instanz erstellen
        fe = featureEngineering()
        
        # Extrahiere Karriere- und Bildungsdaten mit den Handler-Methoden
        career_history = extract_career_data(input_data, fe)
        education_data = extract_education_data(input_data)
        
        if not career_history:
            return {
                "confidence": [0.0],
                "recommendations": ["Keine gültigen Karrieredaten vorhanden"],
                "status": "unbekannt",
                "explanations": []
            }
        
        # Schätze Alterskategorie
        age_category = estimate_age_category(input_data)
        if age_category is None:
            age_category = 0
        
        # Extrahiere Features mit existierender Funktion
        features = extract_additional_features(career_history, education_data, fe, age_category)
        
        # Konstruiere den vollständigen Modellpfad
        model_path = os.path.join(MODEL_BASE_PATH, model_name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modell nicht gefunden unter: {model_path}")
            
        # Lade den Checkpoint
        checkpoint = torch.load(model_path)
        
        # Modell initialisieren
        model = GRUModel(
            input_size=6,  # Anzahl der Features pro Zeitschritt
            hidden_size=64,
            num_layers=3,
            dropout=0.3
        )
        
        # Lade die Modellgewichte
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()

        # Input vorbereiten (nur Positionsfeatures)
        sequence_features = features['career_sequence']
        
        # Konvertiere zu Tensor
        seq_tensor = torch.tensor([
            [float(seq.get(key, 0)) for key in [
                'level', 'branche', 'duration_months', 
                'time_since_start', 'time_until_end', 'is_current']
            ] for seq in sequence_features
        ], dtype=torch.float32).unsqueeze(0)
        
        # Vorhersage machen
        with torch.no_grad():
            pred = model(seq_tensor)
        
        pred_value = float(pred.item())
        
        # Berechne Feature-Importance
        explanations = calculate_feature_importance(model, seq_tensor)
        
        # Interpretation der Vorhersage
        if pred_value > 0.7:
            status = "sehr wahrscheinlich wechselbereit"
            recommendations = [
                "Der Kandidat zeigt starke Anzeichen für einen bevorstehenden Wechsel.",
                "Aktive Ansprache empfohlen.",
                f"Wechselwahrscheinlichkeit: {pred_value:.1%}"
            ]
        elif pred_value > 0.5:
            status = "wahrscheinlich wechselbereit"
            recommendations = [
                "Der Kandidat könnte für einen Wechsel offen sein.",
                "Regelmäßige Kontaktaufnahme empfohlen.",
                f"Wechselwahrscheinlichkeit: {pred_value:.1%}"
            ]
        elif pred_value > 0.3:
            status = "möglicherweise wechselbereit"
            recommendations = [
                "Der Kandidat zeigt keine klaren Anzeichen für einen Wechsel.",
                "Beobachtung der Situation empfohlen.",
                f"Wechselwahrscheinlichkeit: {pred_value:.1%}"
            ]
        else:
            status = "bleibt wahrscheinlich"
            recommendations = [
                "Der Kandidat zeigt wenig Interesse an einem Wechsel.",
                "Längerfristige Beziehungspflege empfohlen.",
                f"Wechselwahrscheinlichkeit: {pred_value:.1%}"
            ]
        
        # Füge die wichtigsten Feature-Erklärungen zu den Empfehlungen hinzu
        if explanations:
            top_features = explanations[:3]
            feature_recommendations = [
                f"• {feature['feature']}: {feature['description']} ({feature['impact_percentage']:.1f}% Einfluss)"
                for feature in top_features
            ]
            recommendations.extend(feature_recommendations)
            
        return {
            "confidence": [pred_value],
            "recommendations": recommendations,
            "status": status,
            "explanations": explanations
        }
    except Exception as e:
        print(f"Fehler bei der Vorhersage: {str(e)}")
        traceback.print_exc()
        return {
            "confidence": [0.0],
            "recommendations": [f"Fehler bei der Vorhersage: {str(e)}"],
            "status": "Fehler",
            "explanations": []
        }