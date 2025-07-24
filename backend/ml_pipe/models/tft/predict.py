from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, GroupNormalizer
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from rapidfuzz import process, fuzz
from datetime import datetime
import sys
import json
import pandas as pd
import torch
import os
import numpy as np
from backend.ml_pipe.explainable_ai.explainer import ModelExplainer
from backend.ml_pipe.models.career_rules import CareerRules

# Lade Konfigurationsdateien relativ zum Skriptpfad
script_dir = os.path.dirname(__file__)
config_path = os.path.join(script_dir, '..', '..', 'data', 'featureEngineering', 'position_level.json')

with open(config_path, "r") as f:
    position_entries = json.load(f)

position_map = {
    entry["position"].lower(): (
        entry["level"], 
        entry["branche"],
        entry.get("durchschnittszeit_tage", 365)
    ) for entry in position_entries
}
all_positions = list(position_map.keys())

# Lade das neue FeatureEngineering
sys.path.insert(0, '/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/')
from backend.ml_pipe.data.featureEngineering.tft.feature_engineering_tft import FeatureEngineering

'''
Helper Functions
'''

def get_branche_num(branche):
    branche_map = {"sales": 1, "engineering": 2, "consulting": 3}
    return branche_map.get(branche, 0)

def check_min_timesteps(df, encoder_length=4, prediction_length=2):
    required = encoder_length + prediction_length
    actual = len(df)
    
    if actual >= required:
        return True, 0
    else:
        return False, required - actual

def append_dummy_rows(df, missing_count):
    df = df.copy()
    last_row = df.iloc[-1].copy()
    
    for i in range(missing_count):
        new_row = last_row.copy()
        new_row["time_idx"] = df["time_idx"].max() + 1 if "time_idx" in df.columns else len(df) + 1
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    return df

'''
Explanation Functions
'''

def get_feature_names():
    return [
        "Work Experience (Days)",
        "Number of Changes",
        "Number of Jobs",
        "Average Job Duration",
        "Highest Degree",
        "Age Category",
        "Position Level",
        "Industry",
        "Position Average Duration",
        "Position ID",
        "Weekday",
        "Weekday (Cyclic)",
        "Month",
        "Month (Cyclic)",
        "Previous Position 1 Level",
        "Previous Position 1 Industry",
        "Previous Position 1 Duration",
        "Previous Position 2 Level",
        "Previous Position 2 Industry",
        "Previous Position 2 Duration",
        "Company Size",
        "Study Field"
    ]

def map_tft_feature_names(feature_name):
    mapping = {
        'berufserfahrung_tage': 'Work Experience',
        'anzahl_wechsel_bisher': 'Number of Changes',
        'anzahl_jobs_bisher': 'Number of Jobs',
        'durchschnittsdauer_jobs': 'Average Job Duration',
        'highest_degree': 'Highest Degree',
        'age_category': 'Age Category',
        'position_level': 'Position Level',
        'position_branche': 'Industry',
        'position_durchschnittszeit': 'Position Average Duration',
        'position_id_numeric': 'Position ID',
        'weekday': 'Weekday',
        'weekday_sin': 'Weekday',
        'weekday_cos': 'Weekday',
        'month': 'Month',
        'month_sin': 'Month',
        'month_cos': 'Month',
        'prev_position_1_level': 'Latest Previous Position Level',
        'prev_position_1_branche': 'Latest Previous Position Industry',
        'prev_position_1_dauer': 'Latest Previous Position Duration',
        'prev_position_2_level': 'Second Latest Previous Position Level',
        'prev_position_2_branche': 'Second Latest Previous Position Industry',
        'prev_position_2_dauer': 'Second Latest Previous Position Duration',
        'company_size': 'Company Size',
        'study_field': 'Study Field',
        'time_idx': 'Career Timeline Position',
        'label_scale': 'Average Job Duration',
        'relative_time_idx': 'Career Progression Stage',
        'encoder_length': 'Historical Data Points'
    }
    return mapping.get(feature_name, feature_name)

def get_feature_description(name):
    descriptions = {
        "Work Experience (Days)": "Cumulative length of professional experience measured in days",
        "Number of Changes": "Total number of job transitions across the candidate's career path",
        "Number of Jobs": "Total count of distinct positions held throughout the career",
        "Average Job Duration": "Average duration (in days) spent in each previous position",
        "Highest Degree": "Highest level of formal education attained",
        "Age Category": "Categorical representation of the candidate's age group ",
        "Position Level": "Level of the current role",
        "Industry": "Economic sector or industry in which the candidate is currently employed",
        "Position Average Duration": "Average tenure observed for similar roles within the dataset",
        "Position ID": "Encoded identifier for the current position type based on role taxonomy",
        "Weekday": "Day of the week on which the most recent job began",
        "Weekday (Cyclic)": "Cyclically encoded weekday to capture temporal patterns for modeling",
        "Month": "Calendar month when the current job started",
        "Month (Cyclic)": "Cyclically encoded month to reflect seasonal effects in career changes",
        "Previous Position 1 Level": "Level of the most recent prior role",
        "Previous Position 1 Industry": "Industry in which the most recent prior role was held",
        "Previous Position 1 Duration": "Duration (in days) of the most recent previous position",
        "Previous Position 2 Level": "Level of the second most recent prior role",
        "Previous Position 2 Industry": "Industry of the second most recent prior role",
        "Previous Position 2 Duration": "Duration (in days) of the second most recent previous position",
        "Company Size": "Size of the employer, typically based on number of employees",
        "Study Field": "Academic discipline or field in which the highest degree was obtained",
        "Career Timeline Position": "Relative point in the candidate's overall career journey",
        "Career Progression Stage": "Abstracted stage of career development, derived from job history and progression patterns",
        "Historical Data Points": "Number of past career events available for analysis"
    }
    return descriptions.get(name, "This feature influences the prediction.")



'''
Feature Engineering Functions
'''

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

def extract_features_from_linkedin_new(data):
    """
    Neue Feature-Extraktion basierend auf dem 22-Feature-System
    """
    rows = []
    
    # Sortiere die Jobs nach Startdatum (älteste zuerst)
    sorted_jobs = sorted(data["workExperience"], 
                        key=lambda x: datetime.strptime(x["startDate"], "%d/%m/%Y"))
    
    total_experience = 0
    job_durations = []
    
    for i, job in enumerate(sorted_jobs):
        start_date = datetime.strptime(job["startDate"], "%d/%m/%Y")
        end_date = datetime.today() if job["endDate"] == "Present" else datetime.strptime(job["endDate"], "%d/%m/%Y")
        
        duration = end_date - start_date
        duration_days = duration.days
        
        if i > 0:
            # Addiere die Dauer des vorherigen Jobs zur Gesamterfahrung
            prev_job = sorted_jobs[i-1]
            prev_start = datetime.strptime(prev_job["startDate"], "%d/%m/%Y")
            prev_end = datetime.today() if prev_job["endDate"] == "Present" else datetime.strptime(prev_job["endDate"], "%d/%m/%Y")
            total_experience += (prev_end - prev_start).days
        
        # Berechne die durchschnittliche Dauer der bisherigen Jobs
        job_durations.append(duration_days)
        avg_duration = sum(job_durations) / len(job_durations) if job_durations else 0
        
        # Konvertiere Zeitstempel zu Unix-Timestamp
        zeitpunkt = start_date.timestamp()
        
        # Schätze Alter und Degree basierend auf Erfahrung
        estimated_age = min(25 + (total_experience // 365), 65)  # Schätzung
        age_category = min((estimated_age - 18) // 10 + 1, 5)   # 1-5 Kategorien
        
        # Degree basierend auf Ausbildung
        degree = 3  # Bachelor als Standard
        if any("master" in edu.get("degree", "").lower() for edu in data.get("education", [])):
            degree = 4
        elif any("phd" in edu.get("degree", "").lower() for edu in data.get("education", [])):
            degree = 5
        
        row = {
            "profile_id": "new_user",
            "aktuelle_position": job["position"],
            "zeitpunkt": zeitpunkt,
            "label": float(duration_days),
            "berufserfahrung_bis_zeitpunkt": float(total_experience),
            "anzahl_wechsel_bisher": i,
            "anzahl_jobs_bisher": i + 1,
            "durchschnittsdauer_bisheriger_jobs": float(avg_duration),
            "highest_degree": degree,
            "age_category": age_category,
            "anzahl_standortwechsel": 0,  # Schätzung
            "study_field": "Informatics",  # Standard
            "company_name": job["company"],
            "company_industry": job.get("companyInformation", {}).get("industry", [""])[0] if job.get("companyInformation", {}).get("industry") else "",
            "company_location": job.get("location", ""),
            "company_size_category": "medium"  # Standard
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

def predict(linkedin_data, model_path=None):
    """Vorhersage der Tage bis zum Wechsel mit TFT-Modell (Regression)."""
    try:
        print("\n=== Starte TFT-Vorhersage ===")
        
        # Extrahiere Features mit dem neuen System
        df_new = extract_features_from_linkedin_new(linkedin_data)
        print(f"Extrahierte Features: {len(df_new)} Zeilen")
        
        # Verwende das neue FeatureEngineering
        feature_engineering = FeatureEngineering()
        
        # Konvertiere zu Dokumenten-Format
        docs = df_new.to_dict('records')
        sequences, labels, positions = feature_engineering.extract_sequences_by_profile(docs, min_seq_len=2)
        
        # === Career Rule: Letzte Position < 6 Monate ===
        career_history = linkedin_data.get('workExperience', [])
        print(f"Career history: {career_history}")
        too_new, months = CareerRules.is_last_position_too_new(career_history, min_months=8)
        print(f"Too new: {too_new}, Months: {months}")
        if too_new:
            # Erstelle SHAP und LIME Erklärungen für die Career Rule
            feature_names = get_feature_names()
            
            # SHAP Erklärung: Aktuelle Position zu 100%
            shap_explanations = [{
                "feature": "duration current position",
                "impact_percentage": 100.0,
                "method": "SHAP",
                "description": "The current position is too new for a change."
            }]
            
            # LIME Erklärung: Aktuelle Position zu 100%
            lime_explanations = [{
                "feature": "duration current position", 
                "impact_percentage": 100.0,
                "method": "LIME",
                "description": "The current position is too new for a change."
            }]
            
            return {
                "confidence": [400],  # 0% Wechselwahrscheinlichkeit
                "recommendations": [
                    "The current position is too new for a change.",
                    f"Months in current position: {months:.1f}"
                ],
                "status": "Very unlikely",
                "shap_explanations": shap_explanations,
                "lime_explanations": lime_explanations,
                "llm_explanation": "Candidate is too new in the current position."
            }
        
        print(f"Sequences shape: {sequences.shape}")
        print(f"Labels shape: {labels.shape}")
        
        # Erstelle DataFrame für Vorhersage
        prediction_data = []
        for i, (seq, label, pos_seq) in enumerate(zip(sequences, labels, positions)):
            for j, (features, position) in enumerate(zip(seq, pos_seq)):
                # Konvertiere PyTorch-Tensoren zu normalen Werten
                features_numeric = [float(f.item()) if hasattr(f, 'item') else float(f) for f in features]
                label_numeric = float(label.item()) if hasattr(label, 'item') else float(label)
                
                # Filtere Padding-Zeilen
                if sum(features_numeric) == 0:
                    continue
                
                prediction_data.append({
                    'profile_id': i,
                    'time_idx': j,
                    'target': label_numeric,
                    'position': position,
                    **{f'feature_{k}': v for k, v in enumerate(features_numeric)}
                })
        
        df_prediction = pd.DataFrame(prediction_data)
        print(f"Prediction DataFrame shape: {df_prediction.shape}")
        
        # Feature-Namen für bessere Interpretierbarkeit
        feature_names = {
            'feature_0': 'berufserfahrung_tage',
            'feature_1': 'anzahl_wechsel_bisher',
            'feature_2': 'anzahl_jobs_bisher',
            'feature_3': 'durchschnittsdauer_jobs',
            'feature_4': 'highest_degree',
            'feature_5': 'age_category',
            'feature_6': 'position_level',
            'feature_7': 'position_branche',
            'feature_8': 'position_durchschnittszeit',
            'feature_9': 'position_id_numeric',
            'feature_10': 'weekday',
            'feature_11': 'weekday_sin',
            'feature_12': 'weekday_cos',
            'feature_13': 'month',
            'feature_14': 'month_sin',
            'feature_15': 'month_cos',
            'feature_16': 'prev_position_1_level',
            'feature_17': 'prev_position_1_branche',
            'feature_18': 'prev_position_1_dauer',
            'feature_19': 'prev_position_2_level',
            'feature_20': 'prev_position_2_branche',
            'feature_21': 'prev_position_2_dauer',
            'feature_22': 'company_size',
            'feature_23': 'study_field'
        }
        
        # Benenne Features um
        df_prediction_renamed = df_prediction.rename(columns=feature_names)
        
        # Prüfe minimale Zeitpunkte
        is_valid, missing = check_min_timesteps(df_prediction_renamed, encoder_length=4, prediction_length=2)
        
        if not is_valid:
            df_prediction_renamed = append_dummy_rows(df_prediction_renamed, missing)
            print(f"Added {missing} dummy timepoints. New length: {len(df_prediction_renamed)}")
        
        # Lade das trainierte Modell
        if model_path is None:
            model_path = "/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/models/tft/saved_models/tft_20250627_133624.ckpt"
        
        print(f"Lade trainiertes Modell: {model_path}")
        tft = TemporalFusionTransformer.load_from_checkpoint(model_path)
        
        # Erstelle TimeSeriesDataSet für Vorhersage
        time_varying_unknown_reals_named = [feature_names[f'feature_{i}'] for i in range(24)]
        
        # Konvertiere Position-Strings zu numerischen IDs für kategorische Variable (wie im Training)
        unique_positions = df_prediction_renamed['position'].unique()
        position_to_id = {pos: i for i, pos in enumerate(unique_positions)}
        df_prediction_renamed['position_id'] = df_prediction_renamed['position'].map(position_to_id)
        df_prediction_renamed['position_id'] = df_prediction_renamed['position_id'].astype(str)
        
        prediction_dataset = TimeSeriesDataSet(
            df_prediction_renamed,
            time_idx="time_idx",
            target="target",
            group_ids=["profile_id"],
            max_encoder_length=4,
            max_prediction_length=2,
            min_encoder_length=2,  # Wie im Training
            min_prediction_length=1,
            time_varying_unknown_reals=time_varying_unknown_reals_named,
            time_varying_known_reals=["time_idx"],
            static_categoricals=["position_id"],  # Wie im Training
            categorical_encoders={
                "profile_id": NaNLabelEncoder(add_nan=True),
                "position_id": NaNLabelEncoder(add_nan=True),
            },
            target_normalizer=GroupNormalizer(groups=["profile_id"], transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True,
        )
        
        # Dataloader für Vorhersage
        prediction_dataloader = prediction_dataset.to_dataloader(train=False, batch_size=1)
        
        # Mache Vorhersage
        print("Mache Vorhersage...")
        output = tft.predict(prediction_dataloader, mode="raw", return_x=True)
        
        # Extrahiere die Vorhersage und wähle intelligente Quantile
        prediction_list = output.output.prediction.tolist()
        print(f"Raw prediction: {prediction_list}")
        
        # Extrahiere verschiedene Quantile aus der letzten Vorhersage
        last_prediction = prediction_list[-1][0]  # Letzte Vorhersage
        
        # Intelligente Quantile-Auswahl
        median_50 = last_prediction[3]  # 50% Quantile
        quantile_75 = last_prediction[4]  # 75% Quantile
        quantile_90 = last_prediction[5]  # 90% Quantile
        quantile_100 = last_prediction[6]  # 100% Quantile
        
        # Wähle das beste Quantile basierend auf Realismus
        if median_50 > 30:  # Wenn Median realistisch ist (>30 Tage)
            tage = median_50
            quantile_used = "50% (Median)"
        elif quantile_75 > 60:  # Wenn 75% Quantile realistisch ist
            tage = quantile_75
            quantile_used = "75%"
        elif quantile_90 > 90:  # Fallback auf 90%
            tage = quantile_90
            quantile_used = "90%"
        else:  # Letzter Fallback auf 100%
            tage = quantile_100
            quantile_used = "100%"
        
        print(f"Predicted days: {tage} (verwendetes Quantile: {quantile_used})")
        
        # Debug: Zeige alle Quantile der letzten Vorhersage
        quantiles = last_prediction
        print(f"Alle Quantile (letzte Vorhersage): 0%={quantiles[0]}, 10%={quantiles[1]}, 25%={quantiles[2]}, 50%={quantiles[3]}, 75%={quantiles[4]}, 90%={quantiles[5]}, 100%={quantiles[6]}")
        
        # Debug: Zeige Quantile-Auswahl
        print(f"Quantile-Auswahl: Median={median_50}, 75%={quantile_75}, 90%={quantile_90}, 100%={quantile_100}")
        print(f"Gewähltes Quantile: {quantile_used} = {tage} Tage")
        
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
        
        # Explainable AI Analyse
        print("\n=== Explainable AI Analyse ===")
        
        # SHAP-Analyse (TFT-interne Berechnung)
        print("Berechne SHAP-Werte...")
        var_weights = output.output.encoder_variables[0, -1]
        feature_names_list = prediction_dataset.time_varying_unknown_reals + [
             "relative_time_idx", "encoder_length"
        ]
        
        weights = var_weights.tolist()
        if isinstance(weights[0], list):
            weights = weights[0]
        
        if len(weights) != len(feature_names_list):
            print("WARNUNG: Länge der weights und feature_names_list stimmt nicht überein!")
            print("weights:", weights)
            print("feature_names_list:", feature_names_list)
        
        # Normalisiere die Gewichte zu Prozentwerten
        total = sum(abs(w) for w in weights)
        norm_weights = [(w / total * 100) if total > 0 else 0 for w in weights]
        
        # Mapping
        shap_explanations = []
        for name, val in zip(feature_names_list, norm_weights):
            mapped_name = map_tft_feature_names(name)
            shap_explanations.append({
                "feature": mapped_name,
                "impact_percentage": float(val),
                "method": "SHAP",
                "description": get_feature_description(mapped_name)
            })
        
        # Sortiere Explanations nach Impact
        shap_explanations = sorted(shap_explanations, key=lambda x: -x['impact_percentage'])
        
        # Liste der technischen Features, die ausblenden werden
        technical_features = ["encoder_length", "relative_time_idx", "target_scale", "target_center", "Career Timeline Position", "Career Progression Stage", "Historical Data Points"]

        # Filtere technische Features
        shap_explanations = [
            e for e in shap_explanations
            if e["feature"] not in technical_features
        ]
        
        # Normalisiere die Gewichte neu
        total = sum(e["impact_percentage"] for e in shap_explanations)
        for e in shap_explanations:
            e["impact_percentage"] = e["impact_percentage"] / total * 100 if total > 0 else 0
        
        # Erstelle SHAP Summary
        if len(shap_explanations) >= 2:
            shap_summary = f"Die Vorhersage wurde hauptsächlich beeinflusst durch {shap_explanations[0]['feature']} und {shap_explanations[1]['feature']}."
        else:
            shap_summary = "Keine SHAP-Erklärung verfügbar."
        
        print(shap_summary)
        
        # LIME wird für TFT-Modelle nicht unterstützt
        print("\nLIME-Analyse wird für TFT-Modelle nicht unterstützt.")
        lime_explanations = []
        lime_summary = "LIME wird für TFT-Modelle nicht unterstützt. Verwende nur SHAP-Erklärungen."
        
        # Debug: Zeige verfügbare Erklärungen
        print(f"Verfügbare SHAP-Erklärungen: {len(shap_explanations)}")
        print(f"Verfügbare LIME-Erklärungen: {len(lime_explanations)}")
        
        print("\n=== TFT-Vorhersage abgeschlossen ===")

        return {
            "confidence": tage,  # Einzelner Wert in Tagen
            "recommendations": [recommendation],
            "status": status,
            "shap_explanations": shap_explanations,
            "shap_summary": shap_summary,
            "lime_explanations": lime_explanations,
            "lime_summary": lime_summary,
            "llm_explanation": ""
        }

    except Exception as e:
        print(f"Fehler bei der TFT-Vorhersage: {str(e)}")
        raise
