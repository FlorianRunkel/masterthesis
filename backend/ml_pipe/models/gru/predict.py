import torch
import os
import glob
import json
import requests
from backend.ml_pipe.data.featureEngineering.featureEngineering import featureEngineering
from backend.ml_pipe.data.linkedInData.handler import extract_career_data, extract_education_data, extract_additional_features, estimate_age_category
from backend.ml_pipe.models.gru.model import GRUModel
import numpy as np
import shap
from groq import Groq

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

def generate_llm_explanation(profile_data, explanations, recommendations, model_name="llama3-8b-8192", max_tokens=256, temperature=0.7):
    prompt = f"""
    Du bist ein erfahrener Recruiting-Experte mit Verständnis für datengetriebene Modelle zur Vorhersage von Wechselbereitschaft.

    Hier sind die Profildaten eines Kandidaten:
    {profile_data}

    Die berechnete Wechselwahrscheinlichkeit beträgt **{recommendations}**.

    Die wichtigsten Einflussfaktoren auf den Score sind:
    {', '.join([f"{e['feature']} ({e['impact_percentage']}%)" for e in explanations[:3]])}.

    Bitte erkläre einem Recruiter **in klarer, kurzer und verständlicher Sprache**, warum diese Merkmale für die Vorhersage entscheidend waren. Beziehe Dich dabei auf das konkrete Profil.

    Gehe auch darauf ein, **warum der Score genau in dieser Höhe liegt** – also ob das Profil Anzeichen für Unzufriedenheit, hohe Dynamik oder stabile Verhältnisse zeigt.

    Antworte ausschließlich in einem kurzen Fließtext – keine Listen, keine Bullet Points und beende mit einem Overall Suggestion.
    """

    try:
        client = Groq(api_key="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")
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Du bist ein hilfreicher Recruiting-Experte."},
                {"role": "user", "content": prompt}
            ],
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print("Groq-Fehler:", e)


def predict(profile_dict, model_path=None, with_llm_explanation=False):
    # Modellpfad bestimmen
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

    # Feature Engineering
    fe = featureEngineering()
    career_history = extract_career_data(profile_dict, fe)
    education_data = extract_education_data(profile_dict)
    age_category = estimate_age_category(profile_dict)
    features = extract_additional_features(career_history, education_data, fe, age_category)

    # Sequenz-Features vorbereiten
    sequence_features = features.get('career_sequence', [])
    if not sequence_features:
        return {
            "confidence": [0.0],
            "recommendations": ["Keine Karriere-Sequenz vorhanden."],
            "status": "unbekannt",
            "explanations": [],
            "llm_explanation": ""
        }

    seq_tensor = torch.tensor([
        [float(seq.get(key, 0)) for key in [
            'level', 'branche', 'duration_months', 
            'time_since_start', 'time_until_end', 'is_current']
        ] for seq in sequence_features
    ], dtype=torch.float32).unsqueeze(0)  # (1, T, 6)

    # Statische Features vorbereiten
    static_features = torch.tensor([[
        features.get('total_positions', 0),
        features.get('company_changes', 0),
        features.get('total_experience_years', 0),
        features.get('highest_degree', 0),
        features.get('age_category', 0),
        features.get('avg_position_duration_months', 0)
    ]], dtype=torch.float32)  # (1, 6)

    # Dummy: Ersetze das durch echte Trainingsdaten!
    background_seq = torch.zeros((10, seq_tensor.shape[1], seq_tensor.shape[2]))  # 10 Sequenzen, gleiche Länge/Dim wie Input
    background_static = torch.zeros((10, static_features.shape[1]))
    background_data = [background_seq, background_static]

    # Modell laden
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model = GRUModel(seq_input_size=6, static_input_size=6)
    model.load_state_dict(checkpoint)
    model.eval()

    # Vorhersage
    with torch.no_grad():
        pred = model(seq_tensor, static_features)
        prob = torch.sigmoid(pred).item()

    # Status und Empfehlungen
    if prob > 0.7:
        status = "sehr wahrscheinlich wechselbereit"
    elif prob > 0.5:
        status = "wahrscheinlich wechselbereit"
    elif prob > 0.3:
        status = "möglicherweise wechselbereit"
    else:
        status = "bleibt wahrscheinlich"

    recommendations = [
        f"Der Kandidat ist {status}.",
        f"Wechselwahrscheinlichkeit: {prob:.1%}"
    ]

    # SHAP-Explainer und Werte berechnen
    explainer = shap.DeepExplainer(model, background_data)
    shap_values = explainer.shap_values([seq_tensor, static_features], check_additivity=False)

    # Für Sequenz-Features (nur als Beispiel)
    mean_shap_seq = np.mean(np.abs(shap_values[0][0]), axis=0).flatten()  # (6,)
    mean_shap_static = np.mean(np.abs(shap_values[1][0]), axis=0)  # (6,)

    # SHAP-Werte zusammenfassen und auf 100% normieren
    all_shap = np.concatenate([mean_shap_seq, mean_shap_static])
    total = np.sum(all_shap)
    if total == 0:
        norm_shap = np.zeros_like(all_shap)
    else:
        norm_shap = (all_shap / total) * 100  # Prozentwerte

    feature_names = [
        "Position Level", "Branche", "Beschäftigungsdauer",
        "Zeit seit Beginn", "Zeit bis Ende", "Aktuelle Position",
        "Total Positions", "Company Changes", "Total Experience Years",
        "Highest Degree", "Age Category", "Avg. Position Duration"
    ]

    explanations = []
    for name, val in zip(feature_names, norm_shap):
        explanations.append({
            "feature": name,
            "impact_percentage": float(val),
            "description": get_feature_description(name)
        })

    # LLM-Erklärung via Groq
    llm_explanation = ""
    if with_llm_explanation:
        llm_explanation = generate_llm_explanation(profile_dict, explanations, recommendations)

    return {
        "confidence": [float(prob)],
        "recommendations": recommendations,
        "status": status,
        "explanations": explanations,
        "llm_explanation": llm_explanation
    }