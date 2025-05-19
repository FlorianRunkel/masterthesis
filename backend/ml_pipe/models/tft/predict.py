import torch
import json
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from backend.ml_pipe.data.featureEngineering.tft.feature_engineering_tft import FeatureEngineering

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
        
        # Berufserfahrungen extrahieren und formatieren
        work_experiences = []
        for exp in profile_json.get('workExperience', []):
            # Datum-Format anpassen
            start_date = exp['startDate']
            if '/' in start_date:
                # Extrahiere nur Monat und Jahr
                parts = start_date.split('/')
                if len(parts) >= 2:
                    # Wenn das Format DD/MM/YYYY ist, nehmen wir den zweiten Teil als Monat
                    if len(parts) == 3:
                        month, year = parts[1], parts[2]
                    else:
                        month, year = parts[0], parts[1]
                    start_date = f"{month}/{year}"
            
            end_date = exp['endDate']
            if end_date != "Present" and '/' in end_date:
                # Extrahiere nur Monat und Jahr
                parts = end_date.split('/')
                if len(parts) >= 2:
                    # Wenn das Format DD/MM/YYYY ist, nehmen wir den zweiten Teil als Monat
                    if len(parts) == 3:
                        month, year = parts[1], parts[2]
                    else:
                        month, year = parts[0], parts[1]
                    end_date = f"{month}/{year}"
            
            # Neue formatierte Erfahrung hinzufügen
            work_experiences.append({
                "position": exp['position'],
                "company": exp['company'],
                "startDate": start_date,
                "endDate": end_date
            })
        
        return {"workExperience": work_experiences}
    
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

def predict_next_job_change(profile_data):
    try:
        if isinstance(profile_data, str):
            profile_data = json.loads(profile_data)
        # 1. Trainingsdataset laden
        training_dataset = load_tft_training_dataset(
            "/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/models/tft/saved_models/training_dataset.pt"
        )
        # 2. Position Mapping laden
        with open("/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/data/featureEngineering/position_level.json", "r") as f:
            position_entries = json.load(f)
        position_map = {entry["position"].lower(): (entry["level"], entry["branche"]) for entry in position_entries}
        all_positions = list(position_map.keys())
        level_map = {
            1: "Entry", 2: "Junior", 3: "Professional", 4: "Senior", 5: "Lead", 6: "Manager", 7: "Director", 8: "C-Level"
        }
        # 3. Feature Engineering Klasse nutzen
        fe = FeatureEngineering(position_map, all_positions, level_map)
        time_points, experiences = fe.prepare_time_points(profile_data)
        df = fe.prepare_prediction_dataframe(time_points)
        if df is None:
            return None
        # 5. Vorhersage-Dataset erstellen
        prediction_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset,
            df,
            predict=True,
            stop_randomization=True,
            target_normalizer=None
        )
        # 6. Modell laden
        model = load_tft_model("/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/models/tft/saved_models/tft_20250515_163355.ckpt")
        # 7. Vorhersage machen
        dataloader = prediction_dataset.to_dataloader(train=False, batch_size=1)
        predictions = model.predict(dataloader)
        # 8. Ergebnisse formatieren
        results = format_prediction_results(predictions)
        return results
    except Exception as e:
        print(f"Fehler bei der Vorhersage: {str(e)}")
        return None

'''
Result & Explanation Functions
'''
def format_prediction_results(predictions):
    results = []
    for i, pred in enumerate(predictions):
        median_tage = float(pred[0])
        untere_schranke_tage = float(pred[1])
        obere_schranke_tage = float(pred[5])
        results.append({
            "tag": i + 1,
            "vorhersage": {
                "median": median_tage,
                "untere_schranke": untere_schranke_tage,
                "obere_schranke": obere_schranke_tage,
                "unsicherheit": obere_schranke_tage - untere_schranke_tage
            }
        })
    return results

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
def predict(profile_data, with_llm_explanation=False):
    try:
        predictions = predict_next_job_change(profile_data)
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