from backend.ml_pipe.data.featureEngineering.xgboost.feature_engineering_xgb import FeatureEngineering
from backend.ml_pipe.data.linkedInData.classification.profileFeaturizer import extract_career_data, extract_education_data, extract_additional_features, estimate_age_category
import numpy as np
import joblib
import os
import glob
import json
import shap
import pickle

from collections import defaultdict
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

with open("/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/models/xgboost/saved_models/position_categories.pkl", "rb") as f:
    position_categories = pickle.load(f)
position_mapping = {name: idx for idx, name in enumerate(position_categories, 1)}

position_encoder = LabelEncoder()
position_encoder.classes_ = np.array(position_categories)

'''
Helper Functions
'''
def parse_profile_data(profile_dict):
    # If profile is stored as JSON string in 'linkedinProfileInformation', parse it!
    if "linkedinProfileInformation" in profile_dict:
        try:
            return json.loads(profile_dict["linkedinProfileInformation"])
        except Exception as e:
            raise ValueError(f"Profile could not be parsed: {e}")
    return profile_dict

def extract_xgb_features(features_dict):
    """
    Extracts flat features from the feature dictionary for XGBoost.
    Returns a list of numerical features.
    """
    return [
        features_dict.get("total_positions", 0),
        features_dict.get("avg_position_duration_months", 0),
        features_dict.get("company_changes", 0),
        features_dict.get("total_experience_years", 0),
        features_dict.get("highest_degree", 0),
        features_dict.get("age_category", 0),
        features_dict.get("location_changes", 0),
        features_dict.get("unique_locations", 0),
        features_dict.get("current_position", {}).get("level", 0),
        features_dict.get("current_position", {}).get("branche", 0),
        features_dict.get("current_position", {}).get("duration_months", 0),
        features_dict.get("current_position", {}).get("time_since_start", 0),
    ]

# Optional: Funktion für die gesamte Datenvorbereitung für XGBoost

def prepare_xgb_data(data):
    """
    Expects a list of dictionaries with 'features' and 'label'.
    Returns X (2D array) and y (1D array) for XGBoost.
    """
    X = [extract_xgb_features(sample['features']) for sample in data]
    y = [sample.get('label', 0) for sample in data]
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

'''
Feature Engineering / Data Processing Functions
'''
def extract_features(profile_dict):
    fe = FeatureEngineering()
    career_history = extract_career_data(profile_dict, fe)
    education_data = extract_education_data(profile_dict)
    age_category = estimate_age_category(profile_dict)
    return extract_additional_features(career_history, education_data, fe, age_category)

def get_xgb_input(features):
    """
    Converts features to XGBoost format.
    """
    # Dictionary for industry mapping
    industry_levels = {
        "bau": 1,
        "consulting": 2,
        "customerservice": 3,
        "design": 4,
        "education": 5,
        "einkauf": 6,
        "engineering": 7,
        "finance": 8,
        "freelance": 9,
        "gesundheit": 10,
        "healthcare": 11,
        "hr": 12,
        "immobilien": 13,
        "it": 14,
        "legal": 15,
        "logistik": 16,
        "marketing": 17,
        "medien": 18,
        "operations": 19,
        "produktion": 20,
        "projektmanagement": 21,
        "research": 22,
        "sales": 23,
        "verwaltung": 24
    }
    position_str = features.get("position", "").lower()

    try:
        position_encoded = position_encoder.transform([position_str])[0]
    except ValueError:
        position_encoded = 0

    # Convert features to correct order
    xgb_features = [
        features["company_changes"],
        features["total_experience_days"],
        features["location_changes"],
        features["average_days_per_position"],
        features["highest_degree"],
        features["position_level"],
        position_encoded,
        industry_levels.get(features["position_branche"], 0),
        features["duration_days"],
        features["position_duration_days"],
        features["age_category"]
    ]
    
    # Convert to numpy array
    X = np.array([xgb_features], dtype=np.float32)
    return X

def extract_career_history_features(career_history, branche_levels, position_mapping):
    durations = [pos.get("duration", 0) for pos in career_history.values()]
    avg_duration = sum(durations) / len(durations) if durations else 0

    branches = [pos.get("branche", "") for pos in career_history.values()]
    branche_changes = sum(1 for i in range(1, len(branches)) if branches[i] != branches[i-1])
    num_unique_branches = len(set(branches))

    positions = [pos.get("position", "") for pos in career_history.values()]
    position_changes = sum(1 for i in range(1, len(positions)) if positions[i] != positions[i-1])
    num_unique_positions = len(set(positions))

    branche_days = defaultdict(int)
    for pos in career_history.values():
        branche_days[pos.get("branche", "")] += pos.get("duration", 0)
    top_branche_days = sorted(branche_days.items(), key=lambda x: x[1], reverse=True)[:3]
    branche1_days = top_branche_days[0][1] if len(top_branche_days) > 0 else 0
    branche2_days = top_branche_days[1][1] if len(top_branche_days) > 1 else 0
    branche3_days = top_branche_days[2][1] if len(top_branche_days) > 2 else 0

    max_level = max([pos.get("level", 0) for pos in career_history.values()] or [0])

    # Optional: weitere Features, z.B. durchschnittlicher Level, etc.

    # Optional: Die letzten N Positionen als numerische Features (hier N=3)
    last_positions = list(career_history.values())
    last_positions_features = []
    for pos in last_positions:
        last_positions_features.append(pos.get("duration", 0))
        last_positions_features.append(branche_levels.get(pos.get("branche", ""), 0))
        last_positions_features.append(pos.get("level", 0))
        last_positions_features.append(position_mapping.get(pos.get("position", ""), 0))

    return [
        avg_duration,
        branche_changes,
        position_changes,
        num_unique_branches,
        num_unique_positions,
        branche1_days,
        branche2_days,
        branche3_days,
        max_level
    ]
'''
Model Functions
'''
def get_latest_model_path(model_dir="/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/models/xgboost/saved_models"):
    model_files = glob.glob(os.path.join(model_dir, "xgboost_model_*.joblib"))
    if not model_files:
        raise FileNotFoundError(f"No model found in directory {model_dir}")
    # Select the most recently modified model
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model

def load_xgb_model(model_path):
    return joblib.load(model_path)

'''
Explanation Functions
'''
def get_feature_names():
    names = [
        "Company Changes",
        "Total Experience (Days)",
        "Location Changes",
        "Highest Degree",
        "Position Level",
        "Age Category",
        "Current Position (encoded)",
        "Current Position Branche (encoded)",
        "Current Position Duration (Days)",
        "Current Position Avg Duration (Days)",
        "Avg Duration All Positions",
        "Branche Changes",
        "Position Changes",
        "Unique Branches",
        "Unique Positions",
        "Top Branche 1 Days",
        "Top Branche 2 Days",
        "Top Branche 3 Days",
        "Max Level"
    ]
    # Für alle Positionen in der Karrierehistorie (z.B. 10)
    for i in range(1, 11):
        names += [
            f"CareerHistory Position {i} Duration",
            f"CareerHistory Position {i} Branche (encoded)",
            f"CareerHistory Position {i} Level",
            f"CareerHistory Position {i} (encoded)"
    ]
    return names

def get_feature_description(name):
    descriptions = {
        "Company Changes": "Number of company changes in the career.",
        "Total Experience (Days)": "Total professional experience in days.",
        "Location Changes": "Number of location changes in the career.",
        "Highest Degree": "Highest educational degree achieved.",
        "Position Level": "Level of the current position.",
        "Age Category": "Age group of the candidate.",
        "Current Position (encoded)": "Numerical encoding of the current position.",
        "Current Position Branche (encoded)": "Numerical encoding of the current position's industry.",
        "Current Position Duration (Days)": "Duration in the current position (days).",
        "Current Position Avg Duration (Days)": "Average duration in the current position (days).",
        "Avg Duration All Positions": "Average duration across all positions.",
        "Branche Changes": "Number of industry changes in the career.",
        "Position Changes": "Number of position title changes in the career.",
        "Unique Branches": "Number of unique industries in the career.",
        "Unique Positions": "Number of unique position titles in the career.",
        "Top Branche 1 Days": "Days spent in the most frequent industry.",
        "Top Branche 2 Days": "Days spent in the second most frequent industry.",
        "Top Branche 3 Days": "Days spent in the third most frequent industry.",
        "Max Level": "Highest position level achieved in the career."
    }
    # Für die Karrierehistorie-Features
    for i in range(1, 11):
        descriptions[f"CareerHistory Position {i} Duration"] = f"Duration (days) in career history position {i}."
        descriptions[f"CareerHistory Position {i} Branche (encoded)"] = f"Industry (encoded) in career history position {i}."
        descriptions[f"CareerHistory Position {i} Level"] = f"Level in career history position {i}."
        descriptions[f"CareerHistory Position {i} (encoded)"] = f"Position (encoded) in career history position {i}."
    return descriptions.get(name, "This feature influences the prediction.")

def get_status(prob):
    if prob > 0.7:
        return "very likely to change"
    elif prob > 0.5:
        return "likely to change"
    elif prob > 0.3:
        return "possibly willing to change"
    else:
        return "likely to stay"

def get_explanations(model, feature_names):
    import numpy as np
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
    return explanations

'''
Main Prediction Function
'''
def predict(profile_dict, model_path=None, with_llm_explanation=False):
    try:
        # Determine model path
        if model_path is None:
            model_path = get_latest_model_path()
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model found at {model_path}")

        # Process profile data
        profile_dict = parse_profile_data(profile_dict)
        print("[DEBUG] Profile data parsed")
        
        # Perform feature engineering
        fe = FeatureEngineering()
        career_history = extract_career_data(profile_dict, fe)
        education_data = extract_education_data(profile_dict)
        age_category = estimate_age_category(profile_dict)
        
        print(f"[DEBUG] Career history found: {len(career_history)} entries")
        
        if not career_history:
            raise ValueError("No career history found")
        
        # Extract features for the last position
        last_position = career_history[0]  # The newest position is the first in the sorted list
        print(f"[DEBUG] Last position: {last_position['position']}")
        
        # Calculate features  
        features = {
            "company_changes": len(set(entry['company'] for entry in career_history)) - 1,
            "total_experience_days": int(sum(entry['duration_months'] * 30.44 for entry in career_history)),
            "location_changes": len(set(entry['location'] for entry in career_history if entry['location'])) - 1 if len(set(entry['location'] for entry in career_history if entry['location'])) > 1 else 0,
            "average_days_per_position": int(sum(entry['duration_months'] * 30.44 for entry in career_history) / len(career_history)),
            "highest_degree": extract_additional_features(career_history, education_data, fe, age_category)['highest_degree'],
            "position_level": last_position['level'],
            "position_branche": last_position['branche'],
            "position": last_position['position'],
            "position_duration": int(last_position['duration_months'] * 30.44),
            "avg_position_duration_days": last_position.get('durchschnittszeit_tage', 0),
            "age_category": age_category,
            "career_history": {}  # wird gleich befüllt
        }
        
        # Karrierehistorie als Dict wie im Training
        career_history_features = {}
        for j, prev_pos in enumerate(reversed(career_history[1:])):
            career_history_features[f"position_{j+1}"] = {
                "duration": int(prev_pos['duration_months'] * 30.44),
                "branche": prev_pos['branche'],
                "level": prev_pos['level'],
                "position": prev_pos['position']
            }
        features["career_history"] = career_history_features

        # Alle numerischen Features extrahieren
        feature_vector = [
            features["company_changes"],
            features["total_experience_days"],
            features["location_changes"],
            features["highest_degree"],
            features["position_level"],
            features["age_category"],
            position_mapping.get(features["position"], 0),
            position_mapping.get(features["position_branche"], 0),
            features["position_duration"],
        ]
        feature_vector.extend(
            extract_career_history_features(features["career_history"], position_mapping, position_mapping)
        )

        # In XGBoost-kompatibles Format bringen
        X = np.array([feature_vector], dtype=np.float32)
        print(f"[DEBUG] XGBoost Input Shape: {X.shape}")
        print(f"[DEBUG] XGBoost Input: {X}")

        # Load model and make prediction
        model = load_xgb_model(model_path)
        prob = model.predict_proba(X)[0]
        print(f"[DEBUG] Probability: {prob}")
        status = get_status(prob[1])
        recommendations = [
            f"The candidate is {status}.",
            f"Change probability: {prob[1]:.1%}"
        ]

        # Generate explanations
        feature_names = get_feature_names()
        explanations = get_explanations(model, feature_names)

        # SHAP-Explainer und Werte berechnen
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Optional: SHAP-Werte ausgeben
        print("[DEBUG] SHAP values:", shap_values)

        result = {
            "confidence": [float(prob[1])],
            "recommendations": recommendations,
            "status": status,
            "explanations": explanations,
            "shap_values": shap_values.tolist(),  # SHAP-Werte als Liste
        }

        if with_llm_explanation:
            result["llm_explanation"] = "LLM explanation is being generated..."  # LLM explanation could be added here

        return result

    except Exception as e:
        print(f"[ERROR] Error in predict: {str(e)}")
        print(f"[ERROR] Error type: {type(e).__name__}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        raise