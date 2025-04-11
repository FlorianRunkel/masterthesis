import torch
import os
from ml_pipe.data.featureEngineering.featureEngineering import featureEngineering
from ml_pipe.models.gru.model import GRUModel
import numpy as np

def preprocess(user_data):
    fe = featureEngineering()
    features = fe.extract_features_from_single_user(user_data)
    print(features)

    if features is None:
        raise ValueError("Nicht genug Daten für Vorhersage")

    # Konvertiere (1, 51) → (T, 3)
    seq_len = features.shape[1] // 3
    reshaped = features.reshape(seq_len, 3)
    return reshaped.tolist()

def predict(data, model_path="ml_pipe/models/gru/saved_models/gru_model_20250410_110554.pt"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Kein Modell gefunden unter {model_path}")

    model = GRUModel(input_size=3, hidden_size=32)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    X = preprocess(data)
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # → [1, seq_len, 3]

    with torch.no_grad():
        prob = model(X_tensor).item()

    status = "wechselbereit" if prob > 0.5 else "bleibt wahrscheinlich"

    return {
        "confidence": [float(round(prob, 2))],
        "recommendations": [f"Kandidat: {status}"]
    }