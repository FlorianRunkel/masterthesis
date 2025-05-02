from backend.ml_pipe.data.featureEngineering.featureEngineering import featureEngineering
from backend.ml_pipe.data.linkedInData.handler import extract_career_data, extract_education_data, extract_additional_features, estimate_age_category
from backend.ml_pipe.data.featureEngineering.feature_engineering_xgb import extract_xgb_features
import numpy as np
import joblib
import os

def preprocess(profile_dict):
    X = np.array([extract_xgb_features(profile_dict['features'])], dtype=np.float32)
    return X

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

def predict(input_data, model_path="/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/models/xgboost/saved_models/xgboost_model_20250502_121211.joblib"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Kein Modell gefunden unter {model_path}")

    # Falls 'features' nicht im Input, führe Feature Engineering durch
    if 'features' not in input_data:
        fe = featureEngineering()
        career_history = extract_career_data(input_data, fe)
        education_data = extract_education_data(input_data)
        age_category = estimate_age_category(input_data)
        features = extract_additional_features(career_history, education_data, fe, age_category)
        input_data = {"features": features}

    model = joblib.load(model_path)
    X = preprocess(input_data)
    prob = model.predict_proba(X)[0]
    status = "sehr wahrscheinlich wechselbereit" if prob[1] > 0.7 else (
        "wahrscheinlich wechselbereit" if prob[1] > 0.5 else (
            "möglicherweise wechselbereit" if prob[1] > 0.3 else "bleibt wahrscheinlich"
        )
    )
    recommendations = [
        f"Der Kandidat ist {status}.",
        f"Wechselwahrscheinlichkeit: {prob[1]:.1%}"
    ]

    # Feature Importance
    feature_names = get_feature_names()
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

    return {
        "confidence": [float(prob[1])],
        "recommendations": recommendations,
        "status": status,
        "explanations": explanations
    }