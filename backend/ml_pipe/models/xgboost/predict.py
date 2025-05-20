from backend.ml_pipe.data.featureEngineering.xgboost.feature_engineering_xgb import FeatureEngineering
from backend.ml_pipe.data.linkedInData.classification.profileFeaturizer import extract_career_data, extract_education_data, extract_additional_features, estimate_age_category
import numpy as np
import joblib
import os
import glob
import json

'''
Helper Functions
'''
def parse_profile_data(profile_dict):
    # Falls das Profil als JSON-String in 'linkedinProfileInformation' steckt, parsen!
    if "linkedinProfileInformation" in profile_dict:
        try:
            return json.loads(profile_dict["linkedinProfileInformation"])
        except Exception as e:
            raise ValueError(f"Profil konnte nicht geparst werden: {e}")
    return profile_dict

def extract_xgb_features(features_dict):
    """
    Extrahiert flache Features aus dem Feature-Dictionary für XGBoost.
    Gibt eine Liste von numerischen Features zurück.
    """
    return [
        features_dict.get("total_positions", 0),
        features_dict.get("avg_position_duration_months", 0),
        features_dict.get("company_changes", 0),
        features_dict.get("total_experience_years", 0),
        features_dict.get("highest_degree", 0),
        features_dict.get("age_category", 0),
        features_dict.get("location_changes", 0),
        features_dict.get("unique_locations", 0),
        features_dict.get("current_position", {}).get("level", 0),
        features_dict.get("current_position", {}).get("branche", 0),
        features_dict.get("current_position", {}).get("duration_months", 0),
        features_dict.get("current_position", {}).get("time_since_start", 0),
    ]

# Optional: Funktion für die gesamte Datenvorbereitung für XGBoost

def prepare_xgb_data(data):
    """
    Erwartet eine Liste von Dictionaries mit 'features' und 'label'.
    Gibt X (2D-Array) und y (1D-Array) für XGBoost zurück.
    """
    X = [extract_xgb_features(sample['features']) for sample in data]
    y = [sample.get('label', 0) for sample in data]
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32) 

'''
Feature Engineering / Data Processing Functions
'''
def extract_features(profile_dict):
    fe = FeatureEngineering()
    career_history = extract_career_data(profile_dict, fe)
    education_data = extract_education_data(profile_dict)
    age_category = estimate_age_category(profile_dict)
    return extract_additional_features(career_history, education_data, fe, age_category)

def get_xgb_input(features):
    xgb_features = extract_xgb_features(features)
    X = np.array([xgb_features], dtype=np.float32)
    return X

'''
Model Functions
'''
def get_latest_model_path(model_dir="/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/models/xgboost/saved_models"):
    model_files = glob.glob(os.path.join(model_dir, "xgboost_model_*.joblib"))
    if not model_files:
        raise FileNotFoundError(f"Kein Modell gefunden im Verzeichnis {model_dir}")
    # Wähle das zuletzt geänderte Modell
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model

def load_xgb_model(model_path):
    return joblib.load(model_path)

'''
Explanation Functions
'''
def get_feature_names():
    return [
        "Anzahl Positionen",
        "Ø Positionsdauer (Monate)",
        "Firmenwechsel",
        "Berufserfahrung (Jahre)",
        "Höchster Abschluss",
        "Alterskategorie",
        "Standortwechsel",
        "Anzahl Standorte",
        "Level aktuelle Position",
        "Branche aktuelle Position",
        "Dauer aktuelle Position",
        "Zeit seit Beginn aktueller Position"
    ]

def get_feature_description(name):
    descriptions = {
        "Anzahl Positionen": "Viele Positionswechsel können auf Wechselbereitschaft hindeuten.",
        "Ø Positionsdauer (Monate)": "Kurze Verweildauer spricht für häufige Wechsel.",
        "Firmenwechsel": "Viele Firmenwechsel deuten auf Flexibilität oder Unzufriedenheit hin.",
        "Berufserfahrung (Jahre)": "Mehr Erfahrung kann die Wechselwahrscheinlichkeit beeinflussen.",
        "Höchster Abschluss": "Ein höherer Abschluss kann die Karrierechancen und Wechselbereitschaft beeinflussen.",
        "Alterskategorie": "Das Alter beeinflusst die Karrierephase und Wechselmotivation.",
        "Standortwechsel": "Viele Standortwechsel zeigen Mobilität.",
        "Anzahl Standorte": "Vielfalt der Arbeitsorte kann auf Flexibilität hindeuten.",
        "Level aktuelle Position": "Das Level der aktuellen Position beeinflusst die Wechselwahrscheinlichkeit.",
        "Branche aktuelle Position": "Die Branche kann die Wechselmotivation beeinflussen.",
        "Dauer aktuelle Position": "Lange Dauer spricht für Stabilität, kurze für Wechselbereitschaft.",
        "Zeit seit Beginn aktueller Position": "Je länger in der aktuellen Position, desto höher oft die Wechselbereitschaft."
    }
    return descriptions.get(name, "Dieses Feature beeinflusst die Vorhersage.")

def get_status(prob):
    if prob > 0.7:
        return "sehr wahrscheinlich wechselbereit"
    elif prob > 0.5:
        return "wahrscheinlich wechselbereit"
    elif prob > 0.3:
        return "möglicherweise wechselbereit"
    else:
        return "bleibt wahrscheinlich"

def get_explanations(model, feature_names):
    import numpy as np
    importances = model.feature_importances_
    total = np.sum(importances)
    explanations = []
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        if imp / total * 100 > 1.0:
            explanations.append({
                "feature": name,
                "impact_percentage": round(float(imp / total * 100), 1),
                "description": get_feature_description(name)
            })
    if not explanations:
        explanations.append({
            "feature": "Gesamtanalyse",
            "impact_percentage": 0.0,
            "description": "Die Vorhersage basiert auf einer Kombination verschiedener Karrierefaktoren."
        })
    return explanations

'''
Main Prediction Function
'''
def predict(profile_dict, model_path=None, with_llm_explanation=False):
    # Modellpfad bestimmen
    if model_path is None:
        model_path = get_latest_model_path()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Kein Modell gefunden unter {model_path}")

    # Profildaten verarbeiten
    profile_dict = parse_profile_data(profile_dict)
    features = extract_features(profile_dict)
    X = get_xgb_input(features)

    # Modell laden und Vorhersage machen
    model = load_xgb_model(model_path)
    prob = model.predict_proba(X)[0]
    status = get_status(prob[1])
    recommendations = [
        f"Der Kandidat ist {status}.",
        f"Wechselwahrscheinlichkeit: {prob[1]:.1%}"
    ]

    # Erklärungen generieren
    feature_names = get_feature_names()
    explanations = get_explanations(model, feature_names)

    return {
        "confidence": [float(prob[1])],
        "recommendations": recommendations,
        "status": status,
        "explanations": explanations,
    }