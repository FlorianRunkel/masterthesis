from ml_pipe.data.featureEngineering.featureEngineering import featureEngineering
import numpy as np
import joblib
import os

def preprocess(documents):
    fe = featureEngineering()
    X = fe.extract_features_from_single_user(documents)

    if X is None:
        raise ValueError("Nicht genug Daten für Vorhersage")

    if not isinstance(X, np.ndarray):
        X = np.array(X, dtype=np.float32)

    return X 

def predict(data, model_path="ml_pipe/models/xgboost/saved_models/xgboost_model_20250410_110704.joblib"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Kein Modell gefunden unter {model_path}")

    model = joblib.load(model_path)
    X = preprocess(data)
    print(X)

    prob = model.predict_proba(X)[0][1]
    status = "wechselbereit" if prob > 0.5 else "bleibt wahrscheinlich"

    return {
        "confidence": [float(round(prob, 2))],
        "recommendations": [f"Kandidat: {status}"]
    }