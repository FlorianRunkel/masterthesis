import torch
import os
import glob
import json
from backend.ml_pipe.data.featureEngineering.gru.featureEngineering_gru import FeatureEngineering
from backend.ml_pipe.data.linkedInData.classification.profileFeaturizer import extract_career_data, extract_education_data, extract_additional_features, estimate_age_category
from backend.ml_pipe.models.gru.model import GRUModel
import numpy as np
import shap

'''
Helper Functions
'''
def get_latest_model_path(model_dir="/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/models/gru/saved_models"):
    model_files = glob.glob(os.path.join(model_dir, "gru_model_*.pt"))
    if not model_files:
        raise FileNotFoundError(f"Kein Modell gefunden im Verzeichnis {model_dir}")
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model

def get_feature_names():
    return [
        "Position Level",
        "Branche",
        "Beschäftigungsdauer",
        "Zeit seit Beginn",
        "Zeit bis Ende",
        "Aktuelle Position",
        "Total Positions",
        "Company Changes",
        "Total Experience Years",
        "Highest Degree",
        "Age Category",
        "Avg. Position Duration"
    ]

def get_feature_description(name):
    descriptions = {
        "Position Level": "Die Hierarchieebene der aktuellen und vorherigen Positionen hat einen signifikanten Einfluss auf die Wechselbereitschaft.",
        "Branche": "Die Branchenzugehörigkeit und deren Entwicklung sind wichtige Indikatoren für potenzielle Wechsel.",
        "Beschäftigungsdauer": "Die Verweildauer in Positionen gibt Aufschluss über das Wechselverhalten.",
        "Zeit seit Beginn": "Die Zeit seit Beginn der aktuellen Position beeinflusst die Wechselwahrscheinlichkeit.",
        "Zeit bis Ende": "Die Dauer bis zum Ende einer Position zeigt Muster im Wechselverhalten.",
        "Aktuelle Position": "Der aktuelle Karrierestatus ist ein wichtiger Indikator für Wechselbereitschaft.",
        "Total Positions": "Die Anzahl der bisherigen Positionen beeinflusst die Wechselwahrscheinlichkeit.",
        "Company Changes": "Die Anzahl der Firmenwechsel ist ein Indikator für Flexibilität oder Unzufriedenheit.",
        "Total Experience Years": "Mehr Erfahrung kann die Wechselwahrscheinlichkeit beeinflussen.",
        "Highest Degree": "Ein höherer Abschluss kann die Karrierechancen und Wechselbereitschaft beeinflussen.",
        "Age Category": "Das Alter beeinflusst die Karrierephase und Wechselmotivation.",
        "Avg. Position Duration": "Kurze Verweildauer spricht für häufige Wechsel."
    }
    return descriptions.get(name, "Dieses Feature beeinflusst die Vorhersage.")

def get_status_and_recommendation(tage):
    if tage < 30:
        return "baldiger Wechsel", "Sehr wahrscheinlicher Jobwechsel innerhalb des nächsten Monats"
    elif tage < 90:
        return "mittelfristig", "Wahrscheinlicher Jobwechsel innerhalb der nächsten 3 Monate"
    elif tage < 180:
        return "später Wechsel", "Möglicher Jobwechsel innerhalb der nächsten 6 Monate"
    else:
        return "langfristig", "Jobwechsel in weiterer Zukunft (> 6 Monate)"

'''
Data Processing Functions
'''
def parse_profile_data(profile_dict):
    """Verarbeitet die eingehenden Profildaten."""
    if "linkedinProfileInformation" in profile_dict:
        try:
            return json.loads(profile_dict["linkedinProfileInformation"])
        except Exception as e:
            raise ValueError(f"Profil konnte nicht geparst werden: {e}")
    return profile_dict

def extract_features(profile_dict):
    """Extrahiert Features aus den Profildaten."""
    fe = FeatureEngineering()
    career_history = extract_career_data(profile_dict, fe)
    education_data = extract_education_data(profile_dict)
    age_category = estimate_age_category(profile_dict)
    return extract_additional_features(career_history, education_data, fe, age_category)

def prepare_sequence_tensor(sequence_features):
    """Bereitet die Sequenz-Features als Tensor vor."""
    return torch.tensor([
        [float(seq.get(key, 0)) for key in [
            'level', 'branche', 'duration_months', 
            'time_since_start', 'time_until_end', 'is_current']
        ] for seq in sequence_features
    ], dtype=torch.float32).unsqueeze(0)  # (1, T, 6)

def prepare_static_tensor(features):
    """Bereitet die statischen Features als Tensor vor."""
    return torch.tensor([[
        features.get('total_positions', 0),
        features.get('company_changes', 0),
        features.get('total_experience_years', 0),
        features.get('highest_degree', 0),
        features.get('age_category', 0),
        features.get('avg_position_duration_months', 0)
    ]], dtype=torch.float32)  # (1, 6)

'''
Model Functions
'''
def load_model(model_path):
    """Lädt das GRU-Modell."""
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = GRUModel(seq_input_size=7)
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def create_background_data(seq_tensor):
    background_seq = torch.zeros((10, seq_tensor.shape[1], seq_tensor.shape[2]))
    return background_seq

def get_prediction_status(prob):
    """Bestimmt den Status basierend auf der Vorhersagewahrscheinlichkeit."""
    if prob > 0.7:
        return "sehr wahrscheinlich wechselbereit"
    elif prob > 0.5:
        return "wahrscheinlich wechselbereit"
    elif prob > 0.3:
        return "möglicherweise wechselbereit"
    return "bleibt wahrscheinlich"

'''
Explanation Functions
'''
def calculate_shap_values(model, background_data, seq_tensor):
    """Berechnet SHAP-Werte für die Erklärung."""
    explainer = shap.DeepExplainer(model, background_data)
    shap_values = explainer.shap_values(seq_tensor, check_additivity=False)
    mean_shap_seq = np.mean(np.abs(shap_values[0][0]), axis=0).flatten()
    total = np.sum(mean_shap_seq)
    return (mean_shap_seq / total) * 100 if total > 0 else np.zeros_like(mean_shap_seq)

def create_explanations(norm_shap):
    """Erstellt die Feature-Erklärungen."""
    feature_names = get_feature_names()
    explanations = []
    for name, val in zip(feature_names, norm_shap):
        explanations.append({
            "feature": name,
            "impact_percentage": float(val),
            "description": get_feature_description(name)
        })
    return explanations

'''
Main Prediction Function
'''
def predict(profile_dict, model_path=None, with_llm_explanation=True):
    """Vorhersage der Tage bis zum Wechsel (Regression)."""
    if model_path is None:
        model_path = get_latest_model_path()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Kein Modell gefunden unter {model_path}")

    fe = FeatureEngineering()
    features, _ = fe.extract_features_and_labels_for_training([profile_dict])
    seq_tensor = features  # (1, 1, 7)

    model = load_model(model_path)

    with torch.no_grad():
        pred = model(seq_tensor)
        tage = pred.item()  # Direkter Regressionswert
        print(tage)

    status, recommendation = get_status_and_recommendation(tage)

    # SHAP-Explanations
    background_data = create_background_data(seq_tensor)
    norm_shap = calculate_shap_values(model, background_data, seq_tensor)
    explanations = create_explanations(norm_shap)

    return {
        "confidence": tage,
        "recommendations": [recommendation],
        "status": status,
        "explanations": explanations,
        "llm_explanation": ""
    }
