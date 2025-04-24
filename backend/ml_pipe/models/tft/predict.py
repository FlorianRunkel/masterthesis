import torch
import os
import sys
import traceback
import numpy as np
import shap
from backend.ml_pipe.data.featureEngineering.featureEngineering import featureEngineering
from backend.ml_pipe.data.linkedInData.handler import extract_career_data, extract_education_data, extract_additional_features, estimate_age_category
from backend.ml_pipe.models.tft.model import TFTModel

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
        seq_tensor, global_tensor, lengths = x
        return self.model((seq_tensor, global_tensor, lengths))

def calculate_feature_importance(model, seq_tensor, global_tensor, lengths):
    """Berechnet Feature-Wichtigkeiten basierend auf den Modelleingaben"""
    try:
        # Erstelle Feature-Namen und deren Beiträge basierend auf den tatsächlichen Daten
        sequence_features = seq_tensor[0].numpy()  # [sequence_length, features]
        global_features = global_tensor[0].numpy()  # [global_features]
        
        # Berechne die durchschnittlichen Werte für Sequenz-Features
        avg_sequence_features = np.mean(np.abs(sequence_features), axis=0)
        
        # Kombiniere alle Feature-Werte
        all_feature_values = np.concatenate([avg_sequence_features, np.abs(global_features)])
        
        # Normalisiere die Werte
        total_importance = np.sum(all_feature_values)
        if total_importance == 0:
            feature_importance = np.zeros_like(all_feature_values)
        else:
            feature_importance = (all_feature_values / total_importance) * 100
        
        # Feature-Namen definieren
        sequence_feature_names = [
            "Position Level",
            "Branche",
            "Beschäftigungsdauer",
            "Zeit seit Beginn",
            "Zeit bis Ende",
            "Lücke zwischen Positionen",
            "Level-Änderung",
            "Interner Wechsel",
            "Standortwechsel",
            "Branchenwechsel",
            "Vorheriges Level",
            "Vorherige Branche",
            "Vorherige Dauer"
        ]
        
        global_feature_names = [
            "Bildungsabschluss",
            "Alterskategorie",
            "Berufserfahrung",
            "Durchschnittliche Positionslücke",
            "Interne Wechsel Rate",
            "Standortwechsel Rate",
            "Branchenwechsel Rate",
            "Durchschnittliche Level-Änderung",
            "Positive Wechsel Rate"
        ]
        
        # Kombiniere alle Feature-Namen
        all_feature_names = sequence_feature_names + global_feature_names
        
        # Erstelle Feature-Erklärungen
        explanations = []
        for name, importance in zip(all_feature_names, feature_importance):
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
        "Lücke zwischen Positionen": "Zeitliche Lücken zwischen Positionen können auf Wechselbereitschaft hinweisen.",
        "Level-Änderung": "Veränderungen in der Hierarchieebene zeigen Karriereentwicklung.",
        "Interner Wechsel": "Die Häufigkeit interner Wechsel gibt Aufschluss über Loyalität.",
        "Standortwechsel": "Die Bereitschaft zu Standortwechseln zeigt Flexibilität.",
        "Branchenwechsel": "Wechsel zwischen Branchen deuten auf Anpassungsfähigkeit hin.",
        "Bildungsabschluss": "Der höchste Bildungsabschluss beeinflusst Karrieremöglichkeiten.",
        "Alterskategorie": "Das Alter und die Karrierephase sind wichtige Faktoren.",
        "Berufserfahrung": "Die Gesamtdauer der Berufserfahrung prägt Wechselentscheidungen.",
        "Durchschnittliche Positionslücke": "Regelmäßige Wechsel können ein Muster aufzeigen.",
        "Interne Wechsel Rate": "Das Verhältnis interner zu externen Wechseln ist aufschlussreich.",
        "Standortwechsel Rate": "Die Häufigkeit von Standortwechseln zeigt Mobilität.",
        "Branchenwechsel Rate": "Häufige Branchenwechsel können auf Flexibilität hinweisen.",
        "Durchschnittliche Level-Änderung": "Der Karrierefortschritt beeinflusst Wechselentscheidungen.",
        "Positive Wechsel Rate": "Aufwärtsbewegungen in der Karriere sind wichtige Indikatoren."
    }
    return descriptions.get(feature_name, "Dieses Feature beeinflusst die Vorhersage.")

def predict(input_data, model_name="career_lstm_20250424_175101.pt"):
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
        
        print(features)
        
        # Konstruiere den vollständigen Modellpfad
        model_path = os.path.join(MODEL_BASE_PATH, model_name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modell nicht gefunden unter: {model_path}")
            
        # Lade den Checkpoint
        checkpoint = torch.load(model_path)
        
        # Extrahiere die Hyperparameter
        hyperparameters = checkpoint.get('hyperparameters', {})
        
        # Modell initialisieren
        model = TFTModel(
            sequence_features=hyperparameters.get('sequence_dim', 13),
            global_features=hyperparameters.get('global_dim', 9),
            hidden_size=hyperparameters.get('hidden_size', 128),
            num_layers=hyperparameters.get('num_layers', 2),
            dropout=hyperparameters.get('dropout', 0.2),
            bidirectional=hyperparameters.get('bidirectional', True)
        )
        
        # Lade die Modellgewichte
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()

        # Input vorbereiten
        sequence_features = features['career_sequence']
        global_features = [
            float(features['highest_degree']),
            float(features['age_category']),
            float(features['total_experience_years']),
            float(features.get('avg_position_gap', 0)),
            float(features.get('internal_moves_ratio', 0)),
            float(features.get('location_changes_ratio', 0)),
            float(features.get('branche_changes_ratio', 0)),
            float(features.get('avg_level_change', 0)),
            float(features.get('positive_moves_ratio', 0.5))
        ]
        
        # Konvertiere zu Tensoren
        seq_tensor = torch.tensor([
            [float(seq.get(key, 0)) for key in [
                'level', 'branche', 'duration_months', 
                'time_since_start', 'time_until_end',
                'gap_months', 'level_change', 'internal_move',
                'location_change', 'branche_change', 'previous_level',
                'previous_branche', 'previous_duration'
            ]] for seq in sequence_features
        ], dtype=torch.float32).unsqueeze(0)
        
        global_tensor = torch.tensor(global_features, dtype=torch.float32).unsqueeze(0)
        lengths = torch.tensor([len(sequence_features)]).long()
        
        # Vorhersage machen
        with torch.no_grad():
            pred = model((seq_tensor, global_tensor, lengths))
        
        pred_value = float(pred.item())
        
        # Berechne Feature-Wichtigkeiten
        explanations = calculate_feature_importance(model, seq_tensor, global_tensor, lengths)
        
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
            top_features = sorted(explanations, key=lambda x: x["impact_percentage"], reverse=True)[:3]
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