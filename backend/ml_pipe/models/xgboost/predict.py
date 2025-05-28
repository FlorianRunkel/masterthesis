from backend.ml_pipe.data.featureEngineering.xgboost.feature_engineering_xgb import FeatureEngineering
from backend.ml_pipe.data.linkedInData.classification.profileFeaturizer import extract_career_data, extract_education_data, extract_additional_features, estimate_age_category
import numpy as np
import joblib
import os
import glob
import json

import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
position_classes = joblib.load("/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/models/xgboost/saved_models/position_categories.pkl")
position_encoder = LabelEncoder()
position_encoder.classes_ = np.array(position_classes)

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
    return [
        "Company Changes",
        "Total Experience (Days)",
        "Location Changes",
        "Average Position Duration (Days)",
        "Highest Degree",
        "Position Level",
        "Position Industry",
        "Current Position Duration (Days)",
        "Average Position Duration (Days)",
        "Age Category"
    ]

def get_feature_description(name):
    descriptions = {
        "Company Changes": "Many company changes may indicate willingness to change.",
        "Total Experience (Days)": "More experience can influence the probability of change.",
        "Location Changes": "Many location changes show mobility.",
        "Average Position Duration (Days)": "Short duration suggests frequent changes.",
        "Highest Degree": "A higher degree can influence career opportunities and willingness to change.",
        "Position Level": "The level of the current position influences the probability of change.",
        "Position Industry": "The industry can influence the motivation to change.",
        "Current Position Duration (Days)": "Long duration suggests stability, short duration suggests willingness to change.",
        "Average Position Duration (Days)": "Average time spent in positions.",
        "Age Category": "Age influences career phase and motivation to change."
    }
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
            "location_changes": len(set(entry['location'] for entry in career_history if entry['location'])) - 1,
            "average_days_per_position": int(sum(entry['duration_months'] * 30.44 for entry in career_history) / len(career_history)),
            "highest_degree": extract_additional_features(career_history, education_data, fe, age_category)['highest_degree'],
            "position_level": last_position['level'],
            "position_branche": last_position['branche'],
            "duration_days": int(last_position['duration_months'] * 30.44),
            "position_duration_days": int(last_position.get('durchschnittszeit_tage', last_position['duration_months'] * 30.44)),
            "age_category": age_category
        }
        
        print("[DEBUG] Features calculated:", features)
        
        # Convert features for XGBoost
        X = get_xgb_input(features)
        print(f"[DEBUG] XGBoost Input Shape: {X.shape}")

        # Load model and make prediction
        model = load_xgb_model(model_path)
        prob = model.predict_proba(X)[0]
        status = get_status(prob[1])
        recommendations = [
            f"The candidate is {status}.",
            f"Change probability: {prob[1]:.1%}"
        ]

        # Generate explanations
        feature_names = get_feature_names()
        explanations = get_explanations(model, feature_names)

        result = {
            "confidence": [float(prob[1])],
            "recommendations": recommendations,
            "status": status,
            "explanations": explanations,
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