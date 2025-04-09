import joblib
import numpy as np
import os

"""
Konvertiert das DataFrame in ein Modell-kompatibles Format
"""
def preprocess(user_data):

    features = np.array([[user_data['experience_years'].iloc[0], 2]], dtype=np.float32)
    return features.reshape(1, -1)

def predict(user_data, model_path="saved_models/xgboost_model_20250409_111910.joblib"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Kein Modell gefunden unter {model_path}")

    model = joblib.load(model_path)
    X = preprocess(user_data)

    # Wahrscheinlichkeit für Klasse 1 = wechselbereit
    prob = model.predict_proba(X)[0][1]
    status = "wechselbereit" if prob > 0.5 else "bleibt wahrscheinlich"

    return {
        "confidence": [round(prob, 2)],
        "recommendations": [f"Der Kandidat ist {status} (Schätzung)"]
    }