from backend.ml_pipe.data.featureEngineering.featureEngineering import featureEngineering
from backend.ml_pipe.data.linkedInData.handler import extract_career_data, extract_education_data, extract_additional_features, estimate_age_category
from backend.ml_pipe.data.featureEngineering.feature_engineering_xgb import extract_xgb_features
import numpy as np
import joblib
import os
import glob
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
import requests
import json


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

def get_latest_model_path(model_dir="/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/models/xgboost/saved_models"):
    model_files = glob.glob(os.path.join(model_dir, "xgboost_model_*.joblib"))
    if not model_files:
        raise FileNotFoundError(f"Kein Modell gefunden im Verzeichnis {model_dir}")
    # Wähle das zuletzt geänderte Modell
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model

def predict(profile_dict, model_path=None):
    if model_path is None:
        model_path = get_latest_model_path()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Kein Modell gefunden unter {model_path}")
    
    # Falls das Profil als JSON-String in 'linkedinProfileInformation' steckt, parsen!
    if "linkedinProfileInformation" in profile_dict:
        try:
            profile_dict = json.loads(profile_dict["linkedinProfileInformation"])
        except Exception as e:
            raise ValueError(f"Profil konnte nicht geparst werden: {e}")
    else:
        profile_dict = profile_dict

    # Dann wie gehabt:
    fe = featureEngineering()
    career_history = extract_career_data(profile_dict, fe)
    education_data = extract_education_data(profile_dict)
    age_category = estimate_age_category(profile_dict)
    features = extract_additional_features(career_history, education_data, fe, age_category)

    # Extrahiere flache XGBoost-Features
    xgb_features = extract_xgb_features(features)
    X = np.array([xgb_features], dtype=np.float32)

    model = joblib.load(model_path)
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

    # Score als Prozentwert für die Erklärung übergeben
    score_percent = round(prob[1] * 100, 1)
    llm_explanation = generate_llm_explanation(profile_dict, explanations, recommendations=recommendations)
    return {
        "confidence": [float(prob[1])],
        "recommendations": recommendations,
        "status": status,
        "explanations": explanations,
        "llm_explanation": llm_explanation
    }

def add_gpt2_text_generation(prompt, max_length=100, num_return_sequences=1, seed=42):
    generator = pipeline('text-generation', model='gpt2')
    set_seed(seed)
    results = generator(prompt, max_length=max_length, num_return_sequences=num_return_sequences)
    return [r['generated_text'] for r in results]

def generate_llm_explanation(profile_data, explanations, recommendations, model_name="llama3-8b-8192", max_tokens=256, temperature=0.7):
    headers = {
        "Authorization": f"Bearer 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",
        "Content-Type": "application/json"
    }
    top_features = ", ".join([f"{e['feature']} ({e['impact_percentage']}%)" for e in explanations[:3]])

    prompt = f"""
    Du bist ein erfahrener Recruiting-Experte mit Verständnis für datengetriebene Modelle zur Vorhersage von Wechselbereitschaft.

    Hier sind die Profildaten eines Kandidaten:
    {profile_data}

    Die berechnete Wechselwahrscheinlichkeit beträgt **{recommendations}%**.

    Die wichtigsten Einflussfaktoren auf den Score sind:
    {top_features}.

    Bitte erkläre einem Recruiter **in klarer, kurzer und verständlicher Sprache**, warum diese Merkmale für die Vorhersage entscheidend waren. Beziehe Dich dabei auf das konkrete Profil.

    Gehe auch darauf ein, **warum der Score genau in dieser Höhe liegt** – also ob das Profil Anzeichen für Unzufriedenheit, hohe Dynamik oder stabile Verhältnisse zeigt.

    Antworte ausschließlich in Fließtext – keine Listen, keine Bullet Points.
    """

    data = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"]