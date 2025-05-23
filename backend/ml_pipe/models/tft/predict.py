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

def predict(linkedin_data, with_llm_explanation=False):
    df_new = extract_features_from_linkedin(linkedin_data)
    #print(df_new)

    training_dataset = torch.load(
        "/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/models/tft/training_dataset.pt"
    )

    is_valid, missing = check_min_timesteps(df_new, encoder_length=4, prediction_length=2)

    if not is_valid:
        df_new = append_dummy_rows(df_new, missing)
        print(f"add {missing} Dummy-Zeitpunkte angehängt. Neue Länge: {len(df_new)}")

    print(df_new)
    # Verwende die Struktur des originalen Trainings-Datasets
    new_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        df_new,
        predict=True,   
        allow_missing_timesteps=True,
        min_encoder_length=4,   # WICHTIG!
        min_prediction_idx=1 
    )

    # Dataloader für Vorhersage
    new_dataloader = new_dataset.to_dataloader(train=False, batch_size=1)

    tft = TemporalFusionTransformer.load_from_checkpoint(
        "/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/models/tft/saved_models/tft_20250522_165442.ckpt"
    )

    prediction = tft.predict(new_dataloader, mode="prediction")
    print(prediction)
    first_value = prediction[0][0].item()
    print(first_value)

    return {
        "confidence": first_value,
        "recommendations": [""],
        "status": "",
        "explanations": "",
        "llm_explanation": ""
    }
