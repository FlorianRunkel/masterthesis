import sys
import json
import pandas as pd
import torch
from datetime import datetime
import sys
sys.path.insert(0, '/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/')

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from rapidfuzz import process, fuzz
from datetime import datetime
import shap

with open("/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/data/featureEngineering/position_level.json", "r") as f:
    position_entries = json.load(f)

position_map = {
    entry["position"].lower(): (
        entry["level"], 
        entry["branche"],
        entry.get("durchschnittszeit_tage", 365)
    ) for entry in position_entries
}
all_positions = list(position_map.keys())

def get_branche_num(branche):
    branche_map = {"sales": 1, "engineering": 2, "consulting": 3}
    return branche_map.get(branche, 0)

def map_position_fuzzy(pos, threshold=30):
    pos_clean = pos.lower().strip()

    # 1. Exakter Match (nach lowercase + strip)
    if pos_clean in position_map:
        level, branche, durchschnittszeit = position_map[pos_clean]
        match = pos_clean
    else:
        # 2. Fuzzy Matching (nur wenn kein exakter Treffer)
        match, score, _ = process.extractOne(pos_clean, all_positions, scorer=fuzz.ratio)
        if score < threshold:
            return (None, None, None, None)
        level, branche, durchschnittszeit = position_map[match]

    return (match, float(level), float(get_branche_num(branche)), float(durchschnittszeit))

def append_dummy_rows(df, missing_count):

    df = df.copy()
    last_row = df.iloc[-1].copy()

    for i in range(missing_count):
        new_row = last_row.copy()
        new_row["time_idx"] = df["time_idx"].max() + 1
        new_row["zeitpunkt"] = df["zeitpunkt"].max() + pd.Timedelta(days=30)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    return df

def check_min_timesteps(df, encoder_length=4, prediction_length=2):

    required = encoder_length + prediction_length
    actual = len(df)

    if actual >= required:
        return True, 0
    else:
        return False, required - actual

def extract_features_from_linkedin(data):
    rows = []

    total_experience = 0
    job_durations = []
    
    # Sortiere die Jobs nach Startdatum (älteste zuerst)
    sorted_jobs = sorted(data["workExperience"], 
                        key=lambda x: datetime.strptime(x["startDate"], "%d/%m/%Y"))
    
    for i, job in enumerate(sorted_jobs):
        start_date = datetime.strptime(job["startDate"], "%d/%m/%Y")
        end_date = datetime.today() if job["endDate"] == "Present" else datetime.strptime(job["endDate"], "%d/%m/%Y")
        
        duration = end_date - start_date  # ➜ timedelta
        duration_days = duration.days     # ➜ int
        
        if i > 0:
            # Addiere die Dauer des vorherigen Jobs zur Gesamterfahrung
            prev_job = sorted_jobs[i-1]
            prev_start = datetime.strptime(prev_job["startDate"], "%d/%m/%Y")
            prev_end = datetime.today() if prev_job["endDate"] == "Present" else datetime.strptime(prev_job["endDate"], "%d/%m/%Y")
            total_experience += (prev_end - prev_start).days
        
        # Berechne die durchschnittliche Dauer der bisherigen Jobs
        job_durations.append(duration_days)
        avg_duration = sum(job_durations) / len(job_durations) if job_durations else 0


        mapped_position, level, branche, durchschnittszeit = map_position_fuzzy(job["position"])

        row = {
            "profile_id": "new_user",
            "zeitpunkt": start_date,
            "label": float(duration_days),
            "berufserfahrung_bis_zeitpunkt": float(total_experience),
            "anzahl_wechsel_bisher": i,
            "anzahl_jobs_bisher": i + 1,
            "durchschnittsdauer_bisheriger_jobs": float(avg_duration),
            "mapped_position": mapped_position, 
            "level":level,
            "branche":branche,
            "durchschnittszeit": durchschnittszeit
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df["zeitpunkt"] = pd.to_datetime(df["zeitpunkt"])
    df = df.sort_values("zeitpunkt")
    df["time_idx"] = range(1, len(df) + 1)
    return df

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

def map_tft_feature_names(feature_name):
    mapping = {
        'label_scale': 'Average Job Duration',
        'relative_time_idx': 'Career Progression Stage',
        'berufserfahrung_bis_zeitpunkt': 'Work Experience',
        'time_idx': 'Career Timeline Position',
        'encoder_length': 'Historical Data Points',
        'durchschnittsdauer_bisheriger_jobs': 'Average Job Duration',
        'anzahl_wechsel_bisher': 'Number of Changes',
        'label_center': 'Average Job Duration',
        'anzahl_jobs_bisher': 'Number of Jobs'
    }
    return mapping.get(feature_name, feature_name)

def get_feature_description(name):
    descriptions = {
        "Work Experience": "Total professional experience accumulated over time",
        "Number of Changes": "Total number of job transitions in career history",
        "Number of Jobs": "Total number of positions held throughout career",
        "Average Job Duration": "Mean duration across all previous positions",
        "Position Level": "Current role's seniority and responsibility level",
        "Industry": "Professional sector and business domain",
        "Average Position Duration": "Typical tenure in similar positions",
        "Career Progression Stage": "Current phase in professional development",
        "Career Timeline Position": "Position in overall career trajectory",
        "Historical Data Points": "Number of career events analyzed"
    }
    return descriptions.get(name, "This feature influences the prediction.")

def predict(linkedin_data, with_llm_explanation=False):
    df_new = extract_features_from_linkedin(linkedin_data)

    training_dataset = torch.load(
        "/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/models/tft/training_dataset.pt"
    )

    is_valid, missing = check_min_timesteps(df_new, encoder_length=4, prediction_length=2)

    if not is_valid:
        df_new = append_dummy_rows(df_new, missing)
        print(f"add {missing} Dummy-Zeitpunkte angehängt. Neue Länge: {len(df_new)}")

    # Verwende die Struktur des originalen Trainings-Datasets
    new_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        df_new,
        predict=True,   
        allow_missing_timesteps=True,
        min_encoder_length=4,
        min_prediction_idx=1 
    )

    # Dataloader für Vorhersage
    new_dataloader = new_dataset.to_dataloader(train=False, batch_size=1)

    tft = TemporalFusionTransformer.load_from_checkpoint(
        "/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/models/tft/saved_models/tft_20250522_165442.ckpt"
    )

    output = tft.predict(new_dataloader, mode="raw", return_x=True)
    
    # Extrahiere die Vorhersage und nimm den Median-Wert (3. Wert) aus der ersten Liste
    prediction_list = output.output.prediction.tolist()
    print(prediction_list)
    # Extrahiere den 3. Wert (Index 2) aus der ersten Liste
    print(prediction_list[0][0][3])
    tage = prediction_list[0][0][3]

    # Extrahiere Feature Importance
    var_weights = output.output.encoder_variables[0, -1]
    feature_names = new_dataset.reals + new_dataset.time_varying_unknown_reals
    
    weights = var_weights.tolist()
    if isinstance(weights[0], list):
        weights = weights[0]
    
    # Normalisiere die Gewichte zu Prozentwerten
    total = sum(abs(w) for w in weights)
    norm_weights = [(w / total * 100) if total > 0 else 0 for w in weights]
    
    # Erstelle Explanations im gleichen Format wie GRU
    explanations = []
    for name, val in zip(feature_names, norm_weights):
        mapped_name = map_tft_feature_names(name)
        explanations.append({
            "feature": mapped_name,
            "impact_percentage": float(val),
            "description": get_feature_description(mapped_name)
        })

    # Sortiere Explanations nach Impact
    explanations = sorted(explanations, key=lambda x: -x['impact_percentage'])
    print(explanations)
    
    # Erstelle SHAP Summary
    if len(explanations) >= 2:
        shap_summary = f"Die Vorhersage wurde hauptsächlich beeinflusst durch {explanations[0]['feature']} und {explanations[1]['feature']}."
    else:
        shap_summary = "Keine SHAP-Erklärung verfügbar."

    # Bestimme Status und Empfehlung
    if tage < 30:
        status = "baldiger Wechsel"
        recommendation = "Sehr wahrscheinlicher Jobwechsel innerhalb des nächsten Monats"
    elif tage < 90:
        status = "mittelfristig"
        recommendation = "Wahrscheinlicher Jobwechsel innerhalb der nächsten 3 Monate"
    elif tage < 180:
        status = "später Wechsel"
        recommendation = "Möglicher Jobwechsel innerhalb der nächsten 6 Monate"
    else:
        status = "langfristig"
        recommendation = "Jobwechsel in weiterer Zukunft (> 6 Monate)"

    return {
        "confidence": tage,  # Einzelner Wert in Tagen
        "recommendations": [recommendation],
        "status": status,
        "explanations": explanations,
        "shap_summary": shap_summary,
        "llm_explanation": ""
    }
