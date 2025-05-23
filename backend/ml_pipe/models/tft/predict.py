import torch
import json
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from backend.ml_pipe.data.featureEngineering.tft.feature_engineering_tft import FeatureEngineering
from datetime import datetime, timedelta
import numpy as np
from backend.ml_pipe.data.linkedInData.timeSeries.profileFeaturizer import process_profile
from rapidfuzz import process, fuzz

'''
Helper Functions
'''
def read_linkedin_profile(profile_path):
    try:
        # Wenn profile_path ein String ist und wie ein JSON aussieht, parsen wir es direkt
        if isinstance(profile_path, str) and profile_path.strip().startswith('{'):
            profile_json = json.loads(profile_path)
        else:
            # Ansonsten lesen wir die CSV-Datei
            df = pd.read_csv(profile_path)
            profile_json = json.loads(df.iloc[0]['profile_data'])
        
        return profile_json
    
    except Exception as e:
        print(f"Fehler beim Einlesen des Profils: {str(e)}")
        return None

'''
Model & Prediction Functions
'''
def load_tft_training_dataset(path):
    return torch.load(path)

def load_tft_model(path):
    return TemporalFusionTransformer.load_from_checkpoint(path)

def get_branche_num(branche):
    branche_map = {"sales": 1, "engineering": 2, "consulting": 3}
    return branche_map.get(branche, 0)

def map_position_fuzzy(pos, position_map, all_positions, threshold=30):
    pos_clean = pos.lower().strip()
    if pos_clean in position_map:
        level, branche, durchschnittszeit = position_map[pos_clean]
        match = pos_clean
        score = 100
    else:
        match, score, _ = process.extractOne(pos_clean, all_positions, scorer=fuzz.ratio)
        if score >= threshold:
            level, branche, durchschnittszeit = position_map[match]
        else:
            return (None, None, None, None)
    return (match, float(level), float(get_branche_num(branche)), float(durchschnittszeit))

def prepare_prediction_data(profile_data):
    # Position-Mapping vorbereiten
    with open("/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/data/dataModule/tft/position_to_idx.json") as f:
        raw_position_list = json.load(f)
    
    position_map = {
        entry["position"].lower(): {
            "level": entry["level"],
            "branche": entry["branche"],
            "durchschnittszeit": entry["durchschnittszeit_tage"]
        }
        for entry in raw_position_list
    }
    
    all_positions = list(position_map.keys())
    
    def map_position_fuzzy(pos, threshold=30):
        pos_clean = pos.lower().strip()
        if pos_clean in position_map:
            info = position_map[pos_clean]
            return (pos_clean, info["level"], info["branche"], info["durchschnittszeit"])
        match, score, _ = process.extractOne(pos_clean, all_positions, scorer=fuzz.ratio)
        if score >= threshold:
            info = position_map[match]
            return (match, info["level"], info["branche"], info["durchschnittszeit"])
        return ("unknown", 0, "unknown", 0)
    
    # Feature Engineering
    rows = []
    total_experience = 0
    job_durations = []
    
    # Sortiere die Jobs nach Startdatum (älteste zuerst)
    sorted_jobs = sorted(profile_data["workExperience"], 
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
        
        # Position mappen
        mapped_position, level, branche, durchschnittszeit = map_position_fuzzy(job["position"])
        
        row = {
            "profile_id": "new_user",
            "zeitpunkt": start_date,
            "aktuelle_position": job["position"],
            "mapped_position": mapped_position,
            "level": level,
            "branche": branche,
            "durchschnittszeit": durchschnittszeit,
            "label": float(duration_days),
            "berufserfahrung_bis_zeitpunkt": float(total_experience),
            "anzahl_wechsel_bisher": i,
            "anzahl_jobs_bisher": i + 1,
            "durchschnittsdauer_bisheriger_jobs": float(avg_duration),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df["zeitpunkt"] = pd.to_datetime(df["zeitpunkt"])
    df = df.sort_values("zeitpunkt")
    df["time_idx"] = range(1, len(df) + 1)
    
    # Dummy-Zeilen anhängen falls nötig
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
        return (actual >= required), required - actual
    
    is_valid, missing = check_min_timesteps(df, encoder_length=4, prediction_length=2)
    if not is_valid:
        df = append_dummy_rows(df, missing)
        print(f"➕ {missing} Dummy-Zeitpunkte angehängt. Neue Länge: {len(df)}")
    
    # Nur die benötigten Spalten behalten
    df = df[[
        "profile_id",
        "time_idx",
        "label",
        "berufserfahrung_bis_zeitpunkt",
        "anzahl_wechsel_bisher",
        "anzahl_jobs_bisher",
        "durchschnittsdauer_bisheriger_jobs",
        "level",
        "branche",
        "durchschnittszeit",
        "mapped_position"
    ]]
    
    return df

def predict_next_job_change(profile_data):
    try:
        if isinstance(profile_data, str):
            profile_data = json.loads(profile_data)
            
        # 1. Trainingsdataset laden
        try:
            training_dataset = torch.load(
                "/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/models/tft/saved_models/training_dataset.pt"
            )
            print("Trainingsdataset erfolgreich geladen")
        except Exception as e:
            print(f"Fehler beim Laden des Trainingsdatasets: {str(e)}")
            return None
            
        # 2. DataFrame für Vorhersage vorbereiten
        try:
            df = prepare_prediction_data(profile_data)
            if df is None:
                print("Keine gültigen Daten für die Vorhersage")
                return None
            print("DataFrame erfolgreich erstellt:")
            print(df)
        except Exception as e:
            print(f"Fehler bei der Vorbereitung des DataFrames: {str(e)}")
            return None
            
        # 3. Vorhersage-Dataset erstellen
        try:
            prediction_dataset = TimeSeriesDataSet.from_dataset(
                training_dataset,
                df,
                predict=True,
            )
            print("Vorhersage-Dataset erfolgreich erstellt")
        except Exception as e:
            print(f"Fehler beim Erstellen des Vorhersage-Datasets: {str(e)}")
            return None
            
        # 4. Modell laden
        try:
            model = TemporalFusionTransformer.load_from_checkpoint(
                "/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/models/tft/saved_models/tft_20250522_095807.ckpt"
            )
            print("Modell erfolgreich geladen")
        except Exception as e:
            print(f"Fehler beim Laden des Modells: {str(e)}")
            return None
            
        # 5. Vorhersage machen
        try:
            dataloader = prediction_dataset.to_dataloader(train=False, batch_size=1)
            raw_predictions, x = predict(dataloader, mode="raw", return_x=True)
            print("Vorhersagen erfolgreich erstellt:")
            print(raw_predictions)
            return raw_predictions
        except Exception as e:
            print(f"Fehler bei der Modellvorhersage: {str(e)}")
            return None
            
    except Exception as e:
        print(f"Unerwarteter Fehler bei der Vorhersage: {str(e)}")
        return None

'''
Result & Explanation Functions
'''
def format_prediction_results(predictions):
    try:
        results = []
        # Wir nehmen nur die erste Vorhersage (für den nächsten Zeitpunkt)
        print(f"Verarbeite Vorhersage: {predictions}")
        
        # Die Vorhersagen sind bereits in Tagen
        median_tage = float(predictions[0])
        untere_schranke_tage = float(predictions[0])  # Verwende den ersten Wert als untere Schranke
        obere_schranke_tage = float(predictions[1])   # Verwende den zweiten Wert als obere Schranke
        
        results.append({
            "tag": 1,  # Wir sagen nur den nächsten Tag vorher
            "vorhersage": {
                "median": median_tage,
                "untere_schranke": untere_schranke_tage,
                "obere_schranke": obere_schranke_tage,
                "unsicherheit": obere_schranke_tage - untere_schranke_tage
            }
        })
        return results
    except Exception as e:
        print(f"Fehler in format_prediction_results: {str(e)}")
        print(f"Vorhersagen-Format: {type(predictions)}")
        print(f"Vorhersagen-Inhalt: {predictions}")
        raise e

def generate_recommendation(median_days):
    if median_days is not None:
        if median_days < 30:
            return "Sehr wahrscheinlicher Jobwechsel innerhalb des nächsten Monats"
        elif median_days < 90:
            return "Wahrscheinlicher Jobwechsel innerhalb der nächsten 3 Monate"
        elif median_days < 180:
            return "Möglicher Jobwechsel innerhalb der nächsten 6 Monate"
        else:
            return "Jobwechsel in weiterer Zukunft (> 6 Monate)"
    return "Keine valide Vorhersage möglich."

def generate_explanations(median_days):
    if median_days is not None:
        return [
            f"Basierend auf Ihrer Berufserfahrung wird ein Jobwechsel in etwa {median_days/30:.1f} Monaten erwartet.",
            f"Die Vorhersage basiert auf dem Medianwert der Modellprognose."
        ]
    else:
        return ["Keine valide Vorhersage möglich."]

'''
Main Prediction Function
'''
def predict(profile, with_llm_explanation=False):
    try:
        profile_data = read_linkedin_profile(profile)
        print(profile_data)
        predictions = predict_next_job_change(profile_data)
        print(predictions)
        if predictions is None:
            return {
                'error': 'Fehler bei der Vorhersage',
                'status': 'error'
            }
        median_days = predictions[0]['vorhersage']['median'] if predictions and 'vorhersage' in predictions[0] else None
        recommendation = generate_recommendation(median_days)
        explanations = generate_explanations(median_days)
        return {
            'confidence': [median_days],
            'recommendations': [recommendation],
            'status': 'success',
            'explanations': explanations,
            'predictions': predictions
        }
    except Exception as e:
        print(f"Fehler bei der Vorhersage: {str(e)}")
        return {
            'error': str(e),
            'status': 'error'
        }