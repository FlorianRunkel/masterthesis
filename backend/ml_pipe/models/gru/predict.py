import torch
import os
import glob
import json
from datetime import datetime
from backend.ml_pipe.data.linkedInData.timeSeries.profileFeaturizer import process_profile, parse_date
from backend.ml_pipe.data.featureEngineering.gru.featureEngineering_gru import FeatureEngineering
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
        "Work Experience",
        "Number of Changes",
        "Number of Jobs",
        "Average Job Duration",
        "Position Level",
        "Industry",
        "Average Position Duration"
    ]

def get_feature_description(name):
    descriptions = {
        "Work Experience": "Total professional experience accumulated over time",
        "Number of Changes": "Total number of job transitions in career history",
        "Number of Jobs": "Total number of positions held throughout career",
        "Average Job Duration": "Mean duration across all previous positions",
        "Position Level": "Current role's seniority and responsibility level",
        "Industry": "Professional sector and business domain",
        "Average Position Duration": "Typical tenure in similar positions"
    }
    return descriptions.get(name, "Dieses Feature beeinflusst die Vorhersage.")

def get_status_and_recommendation(tage):
    if tage < 30:
        return "short-term change", "High probability of job change within the next month"
    elif tage < 90:
        return "medium-term change", "High probability of job change within the next 3 months"
    elif tage < 180:
        return "long-term change", "Possible job change within the next 6 months"
    else:
        return "very long-term change", "Job change in further future (> 6 months)"

'''
Data Processing Functions
'''
def parse_profile_data(profile_dict):
    """Verarbeitet die eingehenden Profildaten."""
    if isinstance(profile_dict, str):
        try:
            return json.loads(profile_dict)
        except:
            pass
    
    # Wenn das Profil direkt als Dict übergeben wurde, packen wir es in das erwartete Format
    if isinstance(profile_dict, dict) and "workExperience" in profile_dict:
        profile_dict = {"linkedinProfileInformation": json.dumps(profile_dict)}
    
    return profile_dict

def prepare_prediction_sample(profile_data):
    """Bereitet ein Sample für die Vorhersage vor."""
    try:
        # Wenn die Daten als String übergeben wurden, parsen wir sie
        if isinstance(profile_data, str):
            try:
                profile_data = json.loads(profile_data)
            except:
                pass

        # Wenn die Daten in linkedinProfileInformation sind, extrahieren wir sie
        if isinstance(profile_data, dict) and "linkedinProfileInformation" in profile_data:
            try:
                profile_data = json.loads(profile_data["linkedinProfileInformation"])
            except:
                pass

        experiences = profile_data.get('workExperience', [])
        if not experiences:
            raise ValueError("Keine Berufserfahrung gefunden")

        # Sortiere Erfahrungen nach Startdatum (neueste zuerst)
        experiences = sorted(
            experiences,
            key=lambda x: parse_date(x.get('startDate', '')) or datetime(1900, 1, 1),
            reverse=True
        )

        # Extrahiere Features für die aktuelle Position
        current_exp = experiences[0]  # Neueste Position
        current_start = parse_date(current_exp.get('startDate', ''))
        current_end = parse_date(current_exp.get('endDate', '')) if current_exp.get('endDate') != 'Present' else datetime.now()

        if not current_start:
            raise ValueError("Kein Startdatum für die aktuelle Position gefunden")

        # Berechne Features
        berufserfahrung_bis_zeitpunkt = 0
        anzahl_wechsel_bisher = 0
        anzahl_jobs_bisher = 0
        durchschnittsdauer_bisheriger_jobs = 0

        # Berechne Berufserfahrung
        for exp in experiences:
            start = parse_date(exp.get('startDate', ''))
            end = parse_date(exp.get('endDate', '')) if exp.get('endDate') != 'Present' else datetime.now()
            
            if start and end:
                berufserfahrung_bis_zeitpunkt += (end - start).days
                anzahl_jobs_bisher += 1
                if exp.get('endDate') != 'Present':
                    anzahl_wechsel_bisher += 1

        # Berechne durchschnittliche Jobdauer
        if anzahl_jobs_bisher > 0:
            durchschnittsdauer_bisheriger_jobs = berufserfahrung_bis_zeitpunkt / anzahl_jobs_bisher

        # Erstelle Sample
        sample = {
            "aktuelle_position": current_exp.get("position", ""),
            "berufserfahrung_bis_zeitpunkt": berufserfahrung_bis_zeitpunkt,
            "anzahl_wechsel_bisher": anzahl_wechsel_bisher,
            "anzahl_jobs_bisher": anzahl_jobs_bisher,
            "durchschnittsdauer_bisheriger_jobs": durchschnittsdauer_bisheriger_jobs
        }

        return [sample]  # Liste mit einem Sample zurückgeben

    except Exception as e:
        print(f"Fehler bei der Profilverarbeitung: {str(e)}")
        return []

def prepare_features(profile_dict):
    """Bereitet die Features für die Vorhersage vor."""
    # Profil in das richtige Format bringen
    samples = prepare_prediction_sample(profile_dict)
    
    if not samples:
        raise ValueError("Keine gültigen Samples aus dem Profil extrahiert")

    print(samples)
    # Nehmen wir das aktuellste Sample (erste Position)
    latest_sample = samples[0]
    
    # FeatureEngineering für das Mapping der Positionen
    fe = FeatureEngineering()
    
    # Features in die richtige Reihenfolge bringen wie im Training
    features = [
        float(latest_sample["berufserfahrung_bis_zeitpunkt"]),
        float(latest_sample["anzahl_wechsel_bisher"]),
        float(latest_sample["anzahl_jobs_bisher"]),
        float(latest_sample["durchschnittsdauer_bisheriger_jobs"]),
    ]
    
    # Position und Branche mappen mit verbessertem Matching
    print(f"\nVerarbeite Position: '{latest_sample['aktuelle_position']}'")
    level, branche, durchschnittszeit = fe.map_position(latest_sample["aktuelle_position"])
    features.append(level)
    features.append(branche)
    features.append(durchschnittszeit)
    
    # Debug-Ausgaben
    print("\nFeature-Werte:")
    print(f"Berufserfahrung: {features[0]:.2f} Tage")
    print(f"Anzahl Wechsel: {features[1]:.0f}")
    print(f"Anzahl Jobs: {features[2]:.0f}")
    print(f"Durchschnittsdauer Jobs: {features[3]:.2f} Tage")
    print(f"Level: {features[4]:.0f}")
    print(f"Branche: {features[5]:.0f} ({get_branche_name(features[5])})")
    print(f"Durchschnittszeit Position: {features[6]:.2f} Tage")
    
    # Tensor erstellen mit der gleichen Form wie im Training
    return torch.tensor([features], dtype=torch.float32).unsqueeze(0)  # (1, 1, 7)

def get_branche_name(branche_num):
    """Konvertiert Branchennummer in Namen."""
    branche_map = {1: "Sales", 2: "Engineering", 3: "Consulting"}
    return branche_map.get(branche_num, "Unbekannt")

'''
Model Functions
'''
def load_model(model_path):
    """Loads the GRU model."""
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Create model with correct dimensions
    model = GRUModel(
        seq_input_size=7,      # Features pro Zeitschritt (level, branche, duration_months, etc.)
        hidden_size=256,       # Größerer Hidden Layer für komplexere Muster
        num_layers=4,          # 2 GRU-Schichten
        dropout=0.5,           # Dropout gegen Overfitting
        lr=0.001              # Lernrate
    )
    
    # Load state dict
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Debug: Print model weights
    print("\nModel weights:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.data.mean().item():.4f} (mean)")
    
    return model

def create_background_data(seq_tensor):
    background_seq = torch.zeros((10, seq_tensor.shape[1], seq_tensor.shape[2]))
    return background_seq

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

def create_shap_summary(explanations):
    # Sortiere nach Impact
    sorted_exp = sorted(explanations, key=lambda x: -x['impact_percentage'])
    if not sorted_exp:
        return "No SHAP explanation available."
    if len(sorted_exp) == 1:
        return f"The prediction was mainly influenced by {sorted_exp[0]['feature']}."
    # Nimm die Top 2 Features
    top = sorted_exp[:2]
    return f"The prediction was mainly influenced by {top[0]['feature']} and {top[1]['feature']}."

'''
Main Prediction Function
'''
def predict(profile_dict, model_path="/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/models/gru/saved_models/gru_model_20250526_173300.pt", with_llm_explanation=True):
    """Vorhersage der Tage bis zum Wechsel (Regression)."""
    if model_path is None:
        model_path = get_latest_model_path()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Kein Modell gefunden unter {model_path}")

    print("\n=== Starte Vorhersage ===")
    
    # Profil verarbeiten
    profile_data = parse_profile_data(profile_dict)
    features = prepare_features(profile_data)

    # Modell laden und Vorhersage machen
    model = load_model(model_path)
    with torch.no_grad():
        # Debug: Zwischenwerte der Schichten ausgeben
        gru_out, _ = model.gru(features)
        print("\nGRU Ausgabe:", gru_out.mean().item())
        
        last_out = gru_out[:, -1, :]  # Nur den letzten Output der Sequenz nehmen
        print("GRU letzter Output:", last_out.mean().item())
        
        pred = model.fc(last_out)
        print("Finale Ausgabe:", pred.mean().item())
        
        tage = max(0, pred.item())  # Stelle sicher, dass die Tage nicht unter 0 sind
        print(f"\nModell-Ausgabe:")
        print(f"Rohausgabe: {pred}")
        print(f"Tage bis zum Wechsel: {tage:.2f}")

    status, recommendation = get_status_and_recommendation(tage)

    # SHAP-Explanations
    background_data = create_background_data(features)
    norm_shap = calculate_shap_values(model, background_data, features)
    print(norm_shap)
    explanations = create_explanations(norm_shap)
    shap_summary = create_shap_summary(explanations)

    print("\n=== Vorhersage abgeschlossen ===")

    return {
        "confidence": tage,  # Bleibt in Tagen
        "recommendations": [recommendation],
        "status": status,
        "explanations": explanations,
        "shap_summary": shap_summary,
        "llm_explanation": ""
    }
