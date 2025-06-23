import torch
import os
import glob
import json
from datetime import datetime
from backend.ml_pipe.data.linkedInData.timeSeries.profileFeaturizer import parse_date
from backend.ml_pipe.data.featureEngineering.gru.featureEngineering_gru import FeatureEngineering
from backend.ml_pipe.models.gru.model import GRUModel
import numpy as np
import shap

'''
Helper Functions
'''
def get_latest_model_path(model_dir=None):
    if model_dir is None:
        # Dynamischen Pfad zum Modell-Ordner erstellen
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(script_dir, "saved_models")

    model_files = glob.glob(os.path.join(model_dir, "gru_model_*.pt"))
    if not model_files:
        raise FileNotFoundError(f"Kein Modell gefunden im Verzeichnis {model_dir}")
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model

def get_feature_names():
    return [
        "job experience total",
        "job changes total",
        "job positions total",
        "job average duration",
        "education highest degree",
        "age category",
        "position level",
        "industry",
        "position average duration",
        "position id"
    ]

def get_feature_description(name):
    descriptions = {
        "job experience total": "Total job experience in days",
        "job changes total": "Total job changes",
        "job positions total": "Total job positions",
        "job average duration": "Average duration of previous jobs in days",
        "education highest degree": "Highest education degree (1-5)",
        "age category": "Age category based on career start (1-5)",
        "position level": "Hierarchy level of current position",
        "industry": "Industry/Industry of current company",
        "position average duration": "Average duration in similar positions",
        "position id": "Unique ID of current position"
    }
    return descriptions.get(name, "This feature influences the prediction.")

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
    """Bereitet die Features für die Vorhersage vor (neues Feature Engineering)."""
    try:
        # Profil in das richtige Format bringen
        samples = prepare_prediction_sample(profile_dict)
        if not samples:
            raise ValueError("Keine gültigen Samples aus dem Profil extrahiert")

        latest_sample = samples[0]
        fe = FeatureEngineering()

        # Hilfsfunktionen für zusätzliche Features
        from backend.ml_pipe.data.linkedInData.timeSeries.profileFeaturizer import estimate_age_category, categorize_company_size

        # highest_degree
        def extract_highest_degree(education_data):
            degree_ranking = {
                'phd': 5,
                'master': 4,
                'bachelor': 3,
                'apprenticeship': 2,
                'other': 1
            }
            highest_degree = 1
            for edu in education_data:
                degree = edu.get('degree', '').lower()
                if 'phd' in degree or 'doktor' in degree:
                    highest_degree = max(highest_degree, degree_ranking['phd'])
                elif 'master' in degree or 'msc' in degree or 'mba' in degree:
                    highest_degree = max(highest_degree, degree_ranking['master'])
                elif 'bachelor' in degree or 'bsc' in degree or 'ba' in degree:
                    highest_degree = max(highest_degree, degree_ranking['bachelor'])
                elif 'apprenticeship' in degree or 'ausbildung' in degree:
                    highest_degree = max(highest_degree, degree_ranking['apprenticeship'])
            return highest_degree

        # anzahl_standortwechsel
        def extract_anzahl_standortwechsel(experiences):
            locations = set()
            for exp in experiences:
                loc = exp.get('location', '').strip().lower()
                if loc:
                    locations.add(loc)
            return len(locations)

        # study_field
        def extract_study_field(education_data):
            for edu in education_data:
                val = edu.get('subjectStudy') or edu.get('fieldOfStudy') or edu.get('degree')
                if val and isinstance(val, str) and val.strip():
                    return val.strip()
            return None

        # company_size_category (als Zahl)
        def map_company_size_category(cat):
            mapping = {'micro': 1, 'small': 2, 'medium': 3, 'large': 4, 'enterprise': 5}
            return mapping.get(str(cat).lower(), 0)

        # Profil-Infos extrahieren
        if isinstance(profile_dict, dict) and "linkedinProfileInformation" in profile_dict:
            try:
                profile_info = json.loads(profile_dict["linkedinProfileInformation"])
            except:
                profile_info = profile_dict
        else:
            profile_info = profile_dict

        experiences = profile_info.get('workExperience', [])
        education_data = profile_info.get('education', [])

        # Features extrahieren mit Fehlerbehandlung
        features = [
            float(latest_sample.get("berufserfahrung_bis_zeitpunkt", 0) or 0),
            float(latest_sample.get("anzahl_wechsel_bisher", 0) or 0),
            float(latest_sample.get("anzahl_jobs_bisher", 0) or 0),
            float(latest_sample.get("durchschnittsdauer_bisheriger_jobs", 0) or 0),
            float(extract_highest_degree(education_data) or 0),
            float(estimate_age_category(profile_info) or 0),
            #float(extract_anzahl_standortwechsel(experiences) or 0),
            #float(fe.get_study_field_num(extract_study_field(education_data)) or 0)
        ]

        # Company Size Feature
        current_exp = experiences[0] if experiences else {}
        company_size_cat = categorize_company_size(
            current_exp.get('employee_count') or 
            (current_exp.get('companyInformation', {}) or {}).get('employee_count')
        )
        #features.append(float(map_company_size_category(company_size_cat) or 0))

        # Position-bezogene Features
        level, branche, durchschnittszeit = fe.map_position(latest_sample.get("aktuelle_position", ""))
        features.extend([
            float(level or 0),
            float(branche or 0),
            float(durchschnittszeit or 0)
        ])

        # Positions-ID
        position_idx = fe.get_position_idx(latest_sample.get("aktuelle_position", ""))
        features.append(float(position_idx or 0))

        # Debug-Ausgabe
        print("\nFeature-Werte:")
        for i, (name, val) in enumerate(zip(get_feature_names(), features)):
            print(f"{name}: {val}")

        N = 2
        career_path_features = []
        used_positions = set()
        count = 0

        # Extrahiere echte Positionswechsel (wie im Training)
        experiences = profile_info.get('workExperience', [])
        experiences = sorted(
            experiences,
            key=lambda x: parse_date(x.get('startDate', '')) or datetime(1900, 1, 1)
        )
        last_position = None
        echte_positionen = []
        for exp in experiences:
            pos = exp.get("position", "")
            if pos != last_position:
                echte_positionen.append({
                    "position": pos,
                    "branche": fe.map_position(pos)[1],
                    "durchschnittsdauer": float(latest_sample.get("durchschnittsdauer_bisheriger_jobs", 0) or 0),
                    "zeitpunkt": parse_date(exp.get("startDate", "")).timestamp() if parse_date(exp.get("startDate", "")) else 0
                })
                last_position = pos

        # Aktueller Zeitpunkt (z.B. Startdatum der aktuellen Position)
        current_time = parse_date(current_exp.get("startDate", "")).timestamp() if parse_date(current_exp.get("startDate", "")) else 0

        for prev in reversed(echte_positionen):
            if prev["zeitpunkt"] < current_time and prev["position"] not in used_positions:
                career_path_features.extend([
                    float(fe.map_position(prev["position"])[0]),  # Level
                    float(prev["branche"]),
                    float(prev["durchschnittsdauer"])
                ])
                used_positions.add(prev["position"])
                count += 1
                if count == N:
                    break
        while count < N:
            career_path_features.extend([0.0, 0.0, 0.0])
            count += 1

        features.extend(career_path_features)

        return torch.tensor([features], dtype=torch.float32).unsqueeze(0)  # (1, 1, 13)

    except Exception as e:
        print(f"Fehler bei der Feature-Extraktion: {str(e)}")
        raise

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
    model = GRUModel(seq_input_size=16, hidden_size=128, num_layers=4, dropout=0.2, lr=0.0003)
    
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
    """Erstellt Hintergrunddaten für SHAP mit der richtigen Dimension (13 Features)."""
    background_seq = torch.zeros((10, seq_tensor.shape[1], 16))  # 12 Features
    return background_seq

'''
Explanation Functions
'''
def calculate_shap_values(model, background_data, seq_tensor):
    """Berechnet SHAP-Werte für die Erklärung."""
    # Modell temporär in eval-Modus setzen
    model.eval()
    
    # Einfache Vorhersage für die Erklärung
    with torch.no_grad():
        # Berechne die Vorhersage für die Eingabedaten
        pred_values = []
        for i in range(seq_tensor.shape[0]):
            out, _ = model(seq_tensor[i:i+1])
            pred_values.append(out.item())
        
        # Berechne die Vorhersage für die Hintergrunddaten
        background_values = []
        for i in range(background_data.shape[0]):
            out, _ = model(background_data[i:i+1])
            background_values.append(out.item())
        
        # Berechne die Differenz zwischen Eingabe- und Hintergrundvorhersagen
        shap_values = np.array(pred_values) - np.mean(background_values)
        
        # Normalisiere die Werte
        abs_values = np.abs(shap_values)
        total = np.sum(abs_values)
        return (abs_values / total) * 100 if total > 0 else np.zeros_like(shap_values)

def create_explanations(norm_shap):
    """Erstellt die Feature-Erklärungen."""
    base_len = len(get_feature_names())
    base_shap = norm_shap[:base_len]
    other_shap = norm_shap[base_len:].sum()

    explanations = []
    for name, val in zip(get_feature_names(), base_shap):
        explanations.append({
            "feature": name,
            "impact_percentage": float(val),
            "description": get_feature_description(name)
        })
    if other_shap > 0:
        explanations.append({
            "feature": "Other",
            "impact_percentage": float(other_shap),
            "description": "All other career path features"
        })

    # Optional: Auf 100% normalisieren
    total = sum(e["impact_percentage"] for e in explanations)
    if total > 0:
        for e in explanations:
            e["impact_percentage"] = e["impact_percentage"] * 100 / total

    return explanations

def create_shap_summary(explanations):
    """Erstellt eine Zusammenfassung aller Feature-Einflüsse."""
    if not explanations:
        return "Keine SHAP-Erklärung verfügbar."
    
    # Sortiere nach Impact
    sorted_exp = sorted(explanations, key=lambda x: -x['impact_percentage'])
    
    # Erstelle eine detaillierte Zusammenfassung für alle Features
    summary_parts = []
    for exp in sorted_exp:
        impact = exp['impact_percentage']
        if impact > 0:  # Nur Features mit positivem Einfluss anzeigen
            summary_parts.append(f"{exp['feature']}: {impact:.1f}%")
    
    if not summary_parts:
        return "Keine signifikanten Feature-Einflüsse gefunden."
    
    return "Feature-Einflüsse auf die Vorhersage:\n" + "\n".join(summary_parts)


import numpy as np
import torch
import joblib

def transform_features_for_gru(profile_data, scaler_path="/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/models/gru/scaler_gru.joblib"):
    """Skaliert und transformiert das Profil für das GRU-Modell."""
    # Features extrahieren
    raw_features = prepare_features(profile_data)
    flat = np.array(raw_features).flatten().reshape(1, -1)

    # Skaler laden
    scaler = joblib.load(scaler_path)
    scaled = scaler.transform(flat)

    # In GRU-kompatiblen Tensor umwandeln: (1, 1, F)
    return torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)


'''
Main Prediction Function
'''
def predict(profile_dict, model_path=None, with_llm_explanation=True):
    """Vorhersage der Tage bis zum Wechsel (Regression)."""
    try:
        # Modellpfad bestimmen
        if model_path is None:
            model_path = get_latest_model_path()
            print(f"\nLade neuestes Modell: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Kein Modell gefunden unter {model_path}")

        print("\n=== Starte Vorhersage ===")
        
        # Profil verarbeiten
        profile_data = parse_profile_data(profile_dict)

        # Skaliertes Input-Feature für GRU erzeugen
        features_tensor = transform_features_for_gru(profile_data)

        # Modell laden
        model = load_model(model_path)
        model.eval()
        
        with torch.no_grad():
            gru_out, _ = model.gru(features_tensor)
            print("\nGRU Ausgabe:", gru_out.mean().item())
            
            context, attention_weights = model.attention(gru_out)
            print("Attention Weights:", attention_weights.mean().item())
            
            pred = model.fc(context)
            print("Finale Ausgabe:", pred.mean().item())
            
            tage = max(0, pred.item())
            print(f"\nModell-Ausgabe:")
            print(f"Rohausgabe: {pred}")
            print(f"Tage bis zum Wechsel: {tage:.2f}")

        status, recommendation = get_status_and_recommendation(tage)

        # SHAP-Analyse
        background_data = create_background_data(features_tensor)
        print(background_data.shape)
        
        # Wrapper für SHAP, damit nur der Output-Tensor zurückgegeben wird
        class GRUModelForSHAP(torch.nn.Module):
            def __init__(self, gru_model_lightning):
                super().__init__()
                # Nur relevante Submodule extrahieren
                self.gru = gru_model_lightning.gru
                self.attention = gru_model_lightning.attention
                self.fc = gru_model_lightning.fc

            def forward(self, x):
                gru_out, _ = self.gru(x)
                context = self.attention(gru_out)

                # Falls attention ein Tuple zurückgibt → entpacken:
                if isinstance(context, tuple):
                    context, _ = context  # nur den Context-Vektor nehmen

                output = self.fc(context)
                return output

        shap_model = GRUModelForSHAP(model)
        explainer = shap.DeepExplainer(shap_model, background_data)
        shap_values = explainer.shap_values(features_tensor, check_additivity=False)
        # shap_values[0] hat die Form (samples, features)
        print(shap_values[0].shape)

        # Normalisiere die SHAP-Werte auf Prozentwerte
        abs_sum = np.abs(shap_values[0]).sum()
        if abs_sum > 0:
            norm_shap = (np.abs(shap_values[0][0]) / abs_sum) * 100
        else:
            norm_shap = np.zeros_like(shap_values[0][0])

        explanations = create_explanations(norm_shap)
        shap_summary = create_shap_summary(explanations)
        print(shap_summary)

        print("\n=== Vorhersage abgeschlossen ===")

        return {
            "confidence": torch.expm1(torch.tensor(tage)).item(),  # Rücktransformation
            "recommendations": [recommendation],
            "status": status,
            "explanations": explanations,
            "shap_summary": shap_summary,
            "llm_explanation": "" if not with_llm_explanation else "LLM Explanation folgt."
        }

    except Exception as e:
        print(f"Fehler bei der Vorhersage: {str(e)}")
        raise
