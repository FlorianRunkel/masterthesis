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
from ml_pipe.explainable_ai.explainer import ModelExplainer
from ml_pipe.models.career_rules import CareerRules

'''
Load configuration files relative to script path
'''
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

'''
Load FeatureEngineering
'''
# Removed hardcoded path for Render compatibility
from ml_pipe.data.featureEngineering.tft.feature_engineering_tft import FeatureEngineering

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
        #"Highest Degree",
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
        #'highest_degree': 'Highest Degree',
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
        #"Highest Degree": "Highest level of formal education attained",
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

    if pos_clean in position_map:
        level, branche, durchschnittszeit = position_map[pos_clean]
        match = pos_clean
    else:
        match, score, _ = process.extractOne(pos_clean, all_positions, scorer=fuzz.ratio)
        if score < threshold:
            return (None, None, None, None)
        level, branche, durchschnittszeit = position_map[match]

    return (match, float(level), float(get_branche_num(branche)), float(durchschnittszeit))

'''
Extract features from linkedin
'''
def extract_features_from_linkedin_new(data):
    rows = []
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
            prev_job = sorted_jobs[i-1]
            prev_start = datetime.strptime(prev_job["startDate"], "%d/%m/%Y")
            prev_end = datetime.today() if prev_job["endDate"] == "Present" else datetime.strptime(prev_job["endDate"], "%d/%m/%Y")
            total_experience += (prev_end - prev_start).days

        job_durations.append(duration_days)
        avg_duration = sum(job_durations) / len(job_durations) if job_durations else 0

        zeitpunkt = start_date.timestamp()

        estimated_age = min(25 + (total_experience // 365), 65)  # Schätzung
        age_category = min((estimated_age - 18) // 10 + 1, 5)   # 1-5 Kategorien

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
            #"highest_degree": degree,
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

'''
Predict
'''
def predict(linkedin_data, model_path=None, preloaded_model=None):
    try:
        print("\n=== Start prediction ===")

        df_new = extract_features_from_linkedin_new(linkedin_data)
        print(f"Extracted features: {len(df_new)} rows")

        feature_engineering = FeatureEngineering()

        docs = df_new.to_dict('records')
        sequences, labels, positions = feature_engineering.extract_sequences_by_profile(docs, min_seq_len=2)

        career_history = linkedin_data.get('workExperience', [])
        print(f"Career history: {career_history}")
        rule_applies, info = CareerRules.check_all_rules(career_history, min_years=6, model="tft")
        if rule_applies:
            return info

        print(f"Sequences shape: {sequences.shape}")
        print(f"Labels shape: {labels.shape}")

        prediction_data = []
        for i, (seq, label, pos_seq) in enumerate(zip(sequences, labels, positions)):
            for j, (features, position) in enumerate(zip(seq, pos_seq)):
                features_numeric = [float(f.item()) if hasattr(f, 'item') else float(f) for f in features]
                label_numeric = float(label.item()) if hasattr(label, 'item') else float(label)

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

        feature_names = {
            'feature_0': 'berufserfahrung_tage',
            'feature_1': 'anzahl_wechsel_bisher',
            'feature_2': 'anzahl_jobs_bisher',
            'feature_3': 'durchschnittsdauer_jobs',
            #'feature_4': 'highest_degree',  # Hinzugefügt für Kompatibilität
            'feature_4': 'age_category',
            'feature_5': 'position_level',
            'feature_6': 'position_branche',
            'feature_7': 'position_durchschnittszeit',
            'feature_8': 'position_id_numeric',
            'feature_9': 'weekday',
            'feature_10': 'weekday_sin',
            'feature_11': 'weekday_cos',
            'feature_12': 'month',
            'feature_13': 'month_sin',
            'feature_14': 'month_cos',
            'feature_15': 'prev_position_1_level',
            'feature_16': 'prev_position_1_branche',
            'feature_17': 'prev_position_1_dauer',
            'feature_18': 'prev_position_2_level',
            'feature_19': 'prev_position_2_branche',
            'feature_20': 'prev_position_2_dauer',
            'feature_21': 'company_size',
            'feature_22': 'study_field'
        }

        df_prediction_renamed = df_prediction.rename(columns=feature_names)

        is_valid, missing = check_min_timesteps(df_prediction_renamed, encoder_length=4, prediction_length=2)

        if not is_valid:
            df_prediction_renamed = append_dummy_rows(df_prediction_renamed, missing)
            print(f"Added {missing} dummy timepoints. New length: {len(df_prediction_renamed)}")

        # Use preloaded model if available, otherwise load from file
        if preloaded_model is not None:
            print("Using preloaded TFT model from cache")
            tft = preloaded_model
        else:
            if model_path is None:
                model_path = "ml_pipe/models/tft/saved_models/tft_optimized_20250808_122135.ckpt"
            print(f"Loading TFT model from file: {model_path}")
            tft = TemporalFusionTransformer.load_from_checkpoint(model_path)

        time_varying_unknown_reals_named = [feature_names[f'feature_{i}'] for i in range(23) if f'feature_{i}' in feature_names]

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
            min_encoder_length=2,
            min_prediction_length=1,
            time_varying_unknown_reals=time_varying_unknown_reals_named,
            time_varying_known_reals=["time_idx"],
            static_categoricals=["position_id"],
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

        prediction_dataloader = prediction_dataset.to_dataloader(train=False, batch_size=1)

        print("Make prediction...")
        output = tft.predict(prediction_dataloader, mode="raw", return_x=True)

        prediction_list = output.output.prediction.tolist()
        print(f"Raw prediction: {prediction_list}")

        last_prediction = prediction_list[-1][0]

        median_50 = last_prediction[3]
        quantile_75 = last_prediction[4]
        quantile_90 = last_prediction[5]
        quantile_100 = last_prediction[6]

        if median_50 > 30:
            tage = median_50
            quantile_used = "50% (Median)"
        elif quantile_75 > 60:
            tage = quantile_75
            quantile_used = "75%"
        elif quantile_90 > 90:
            tage = quantile_90
            quantile_used = "90%"
        else:
            tage = quantile_100
            quantile_used = "100%"

        print(f"Predicted days: {tage} (used quantile: {quantile_used})")

        quantiles = last_prediction
        print(f"Alle Quantile (letzte Vorhersage): 0%={quantiles[0]}, 10%={quantiles[1]}, 25%={quantiles[2]}, 50%={quantiles[3]}, 75%={quantiles[4]}, 90%={quantiles[5]}, 100%={quantiles[6]}")

        print(f"Quantile-Auswahl: Median={median_50}, 75%={quantile_75}, 90%={quantile_90}, 100%={quantile_100}")
        print(f"Gewähltes Quantile: {quantile_used} = {tage} Tage")

        if tage < 30:
            status = "short-term change"
            recommendation = "High probability of job change within the next month"
        elif tage < 90:
            status = "medium-term change"
            recommendation = "High probability of job change within the next 3 months"
        elif tage < 180:
            status = "long-term change"
            recommendation = "High probability of job change within the next 6 months"
        else:
            status = "long-term change"
            recommendation = "High probability of job change in the future (> 6 months)"

        print("\n=== Explainable AI Analyse ===")

        print("Calculate SHAP values...")
        var_weights = output.output.encoder_variables[0, -1]
        feature_names_list = prediction_dataset.time_varying_unknown_reals + [
             "relative_time_idx", "encoder_length"
        ]

        weights = var_weights.tolist()
        if isinstance(weights[0], list):
            weights = weights[0]

        if len(weights) != len(feature_names_list):
            print("WARNING: Length of weights and feature_names_list does not match!")
            print("weights:", weights)
            print("feature_names_list:", feature_names_list)

        total = sum(abs(w) for w in weights)
        norm_weights = [(w / total * 100) if total > 0 else 0 for w in weights]

        shap_explanations = []
        for name, val in zip(feature_names_list, norm_weights):
            mapped_name = map_tft_feature_names(name)
            shap_explanations.append({
                "feature": mapped_name,
                "impact_percentage": float(val),
                "method": "SHAP",
                "description": get_feature_description(mapped_name)
            })

        shap_explanations = sorted(shap_explanations, key=lambda x: -x['impact_percentage'])

        technical_features = ["encoder_length", "relative_time_idx", "target_scale", "target_center", "Career Timeline Position", "Career Progression Stage", "Historical Data Points"]

        shap_explanations = [
            e for e in shap_explanations
            if e["feature"] not in technical_features
        ]

        total = sum(e["impact_percentage"] for e in shap_explanations)
        for e in shap_explanations:
            e["impact_percentage"] = e["impact_percentage"] / total * 100 if total > 0 else 0

        if len(shap_explanations) >= 2:
            shap_summary = f"Die Vorhersage wurde hauptsächlich beeinflusst durch {shap_explanations[0]['feature']} und {shap_explanations[1]['feature']}."
        else:
            shap_summary = "Keine SHAP-Erklärung verfügbar."

        print(shap_summary)

        print("\nLIME-Analyse wird für TFT-Modelle nicht unterstützt.")
        lime_explanations = []
        lime_summary = "LIME wird für TFT-Modelle nicht unterstützt. Verwende nur SHAP-Erklärungen."

        print(f"Available SHAP explanations: {len(shap_explanations)}")
        print(f"Available LIME explanations: {len(lime_explanations)}")

        print("\n=== Prediction completed ===")

        return {
            "confidence": tage,
            "recommendations": [recommendation],
            "status": status,
            "shap_explanations": shap_explanations,
            "shap_summary": shap_summary,
            "lime_explanations": lime_explanations,
            "lime_summary": lime_summary,
            "llm_explanation": ""
        }

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        raise
