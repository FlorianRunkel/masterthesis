import torch
import os
from ml_pipe.data.featureEngineering.featureEngineering import featureEngineering
from ml_pipe.models.tft.model import TFTModel

def preprocess(documents, input_size=2):
    fe = featureEngineering()
    X = fe.extract_features_from_single_user(documents)

    if X is None:
        raise ValueError("Nicht genug Daten für Vorhersage")

    return torch.tensor(X, dtype=torch.float32).view(1, -1, input_size)

def predict(data, model_path="ml_pipe/models/tft/saved_models/tft_model_20250409_105005.pt"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Kein Modell gefunden unter {model_path}")

    model = TFTModel(input_size=2, hidden_size=32)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    X = preprocess(data)

    with torch.no_grad():
        prob = model(X).item()

    status = "wechselbereit" if prob > 0.5 else "bleibt wahrscheinlich"

    return {
        "confidence": [float(round(prob, 2))],
        "recommendations": [f"Kandidat: {status}"]
    }