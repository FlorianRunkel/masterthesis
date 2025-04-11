import torch
import os
from ml_pipe.data.featureEngineering.featureEngineering import featureEngineering
from ml_pipe.models.tft.model import TFTModel
from datetime import datetime
import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess(user_data):
    try:
        # Log input data structure
        logger.info(f"Input data type: {type(user_data)}")
        logger.info(f"Input data keys: {user_data.keys() if isinstance(user_data, dict) else 'Not a dictionary'}")
        
        # Handle 'Present' date
        if isinstance(user_data, dict) and 'career_history' in user_data:
            logger.info(f"Career history entries: {len(user_data['career_history'])}")
            
            for job in user_data['career_history']:
                # Convert 'Present' to today's date
                if job.get('endDate') == 'Present':
                    today = datetime.now().strftime('%d/%m/%Y')
                    logger.info(f"Converting 'Present' to today's date: {today}")
                    job['endDate'] = today
        
        # Create feature engineering instance
        fe = featureEngineering()
        
        # Extract features
        logger.info("Starting feature extraction...")
        features = fe.extract_features_from_single_user(user_data)
        logger.info(f"Feature extraction completed: {features is not None}")
        
        if features is None:
            raise ValueError("Feature extraction returned None - insufficient data")
        
        logger.info(f"Features shape: {features.shape if hasattr(features, 'shape') else 'No shape attribute'}")
        
        # Reshape features
        seq_len = features.shape[1] // 3
        reshaped = features.reshape(seq_len, 3)
        logger.info(f"Reshaped features: {reshaped.shape}")
        
        return reshaped.tolist()
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        logger.error(traceback.format_exc())
        raise ValueError(f"Data preprocessing failed: {str(e)}")


def predict(data, model_path="ml_pipe/models/tft/saved_models/tft_model_20250410_164104.pt"):
    try:
        logger.info("Starting prediction...")
        
        # Check model path
        if not os.path.exists(model_path):
            logger.error(f"Model not found at path: {model_path}")
            raise FileNotFoundError(f"Kein Modell gefunden unter {model_path}")
        
        logger.info("Loading model...")
        # Erstelle Modell mit gleicher Architektur wie beim Training
        model = TFTModel(input_size=3, hidden_size=32)
        
        # Lade den state_dict
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Überprüfe auf Schlüsselunterschiede zwischen Modell und state_dict
        model_keys = set(model.state_dict().keys())
        loaded_keys = set(state_dict.keys())
        
        missing_keys = model_keys - loaded_keys
        unexpected_keys = loaded_keys - model_keys
        
        if missing_keys or unexpected_keys:
            logger.warning(f"Modell-Struktur unterscheidet sich:")
            logger.warning(f"Fehlende Schlüssel: {missing_keys}")
            logger.warning(f"Unerwartete Schlüssel: {unexpected_keys}")
            
            # Versuche eine Schlüsselzuordnung, falls möglich
            new_state_dict = {}
            for old_key, value in state_dict.items():
                # Beispiel für Zuordnungen basierend auf den Fehlern
                if old_key == "input_projection.weight" and "input_projection.0.weight" in model_keys:
                    new_state_dict["input_projection.0.weight"] = value
                elif old_key == "input_projection.bias" and "input_projection.0.bias" in model_keys:
                    new_state_dict["input_projection.0.bias"] = value
                elif old_key in model_keys:  # Schlüssel stimmen überein
                    new_state_dict[old_key] = value
            
            # Versuche, das angepasste state_dict zu laden
            if new_state_dict:
                logger.info("Versuche angepasstes state_dict zu laden...")
                model.load_state_dict(new_state_dict, strict=False)
            else:
                logger.warning("Keine Anpassung möglich, lade mit strict=False...")
                model.load_state_dict(state_dict, strict=False)
        else:
            # Normales Laden, wenn die Schlüssel übereinstimmen
            model.load_state_dict(state_dict)
            
        model.eval()
        logger.info("Model loaded successfully")
        
        logger.info("Preprocessing data...")
        X = preprocess(data)
        logger.info("Data preprocessing completed")
        
        logger.info("Converting to tensor...")
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # → [1, seq_len, 3]
        logger.info(f"Input tensor shape: {X_tensor.shape}")
        
        logger.info("Running inference...")
        with torch.no_grad():
            try:
                prob = model(X_tensor).item()
                logger.info(f"Prediction result: {prob}")
                
                status = "wechselbereit" if prob > 0.5 else "bleibt wahrscheinlich"
                
                result = {
                    "confidence": [float(round(prob, 2))],
                    "recommendations": [f"Kandidat: {status}"]
                }
                logger.info(f"Returning result: {result}")
                
                return result
            except Exception as inference_error:
                logger.error(f"Inference error: {str(inference_error)}")
                logger.error(traceback.format_exc())
                return {
                    "error": f"Fehler bei der Inferenz: {str(inference_error)}",
                    "confidence": [0.0],
                    "recommendations": ["Fehler bei der Vorhersage"]
                }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        # Return a structured error response instead of raising an exception
        return {
            "error": str(e),
            "confidence": [0.0],
            "recommendations": ["Fehler bei der Vorhersage"]
        }