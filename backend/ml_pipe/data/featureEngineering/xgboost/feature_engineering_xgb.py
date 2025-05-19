import numpy as np

def extract_xgb_features(features_dict):
    """
    Extrahiert flache Features aus dem Feature-Dictionary für XGBoost.
    Gibt eine Liste von numerischen Features zurück.
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
    Erwartet eine Liste von Dictionaries mit 'features' und 'label'.
    Gibt X (2D-Array) und y (1D-Array) für XGBoost zurück.
    """
    X = [extract_xgb_features(sample['features']) for sample in data]
    y = [sample.get('label', 0) for sample in data]
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32) 