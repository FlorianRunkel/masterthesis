import os
import sys
import logging
from flask import Flask
from flask_cors import CORS

# Im Render-Verzeichnis sind wir bereits im backend/ Ordner

from config import Config
from api_handlers.candidates.handler import candidates_bp
from api_handlers.linkedin.handler import linkedin_bp
from api_handlers.prediction.handler import prediction_bp
from api_handlers.user_management.handler import user_management_bp
from api_handlers.feedback.handler import feedback_bp

'''
Create and configure the backend application

OPTIMIZED GUNICORN COMMAND FOR RENDER:
gunicorn app:app --bind 0.0.0.0:$PORT --workers 3 --timeout 120 --worker-class gevent --worker-connections 1000 --preload --max-requests 1000 --max-requests-jitter 100
'''

# Global model cache for lazy loading
"""
This is a global model cache for lazy loading.
It is used to load the models only when needed.
It is also used to keep the models in memory for faster predictions.
"""
loaded_models = {}
def load_model_lazy(model_type):
    global loaded_models
    if model_type in loaded_models:
        app.logger.info(f"Using cached {model_type.upper()} model")
        return loaded_models[model_type]

    if len(loaded_models) >= 2:
        models_to_remove = [k for k in loaded_models.keys() if k != 'xgboost']
        if models_to_remove:
            oldest_model = models_to_remove[0]
            del loaded_models[oldest_model]
            app.logger.info(f"Removed {oldest_model.upper()} from cache to save RAM (keeping XGBoost)")

    try:
        app.logger.info(f"Loading {model_type.upper()} model on demand...")

        if model_type == 'tft':
            from pytorch_forecasting import TemporalFusionTransformer
            model_path = "/app/backend/ml_pipe/models/tft/saved_models/tft_optimized_20250808_122135.ckpt"
            model = TemporalFusionTransformer.load_from_checkpoint(model_path)

        elif model_type == 'gru':
            from ml_pipe.models.gru.predict import load_model
            model_path = "/app/backend/ml_pipe/models/gru/saved_models/gru_model_20250807_184702.pt"
            model = load_model(model_path)

        elif model_type == 'xgboost':
            from ml_pipe.models.xgboost.predict import load_xgb_model
            model_path = "/app/backend/ml_pipe/models/xgboost/saved_models/xgboost_model_20250806_151510.joblib"
            model = load_xgb_model(model_path)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

        loaded_models[model_type] = model
        app.logger.info(f"{model_type.upper()} model loaded successfully! Cache size: {len(loaded_models)}")
        return model

    except Exception as e:
        app.logger.error(f"Error loading {model_type} model: {str(e)}")
        if model_type != 'xgboost' and 'xgboost' in loaded_models:
            app.logger.warning(f"Falling back to XGBoost model due to {model_type} loading error")
            return loaded_models['xgboost']
        raise

"""
Preload XGBoost model at startup for fastest predictions
"""
def preload_xgboost_on_startup():
    try:
        app.logger.info("Preloading XGBoost model for fastest predictions...")
        xgb_model = load_model_lazy('xgboost')
        app.logger.info("XGBoost model preloaded successfully!")
        return True
    except Exception as e:
        app.logger.error(f"Failed to preload XGBoost: {str(e)}")
        app.logger.warning("XGBoost will be loaded on first request (slower)")
        return False

"""
Create the Flask app
"""
def create_app():

    app = Flask(__name__,
            template_folder=Config.TEMPLATE_DIR,
            static_folder=Config.STATIC_DIR)

    app.logger.setLevel(logging.INFO)

    origins = [
        "https://masterthesis-igbq.onrender.com",
        "https://masterthesis-backend.onrender.com",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    CORS(app,
         origins=origins,
         methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
         allow_headers=["Content-Type", "Authorization", "X-User-Uid"],
         supports_credentials=True)

    app.register_blueprint(candidates_bp)
    app.register_blueprint(linkedin_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(user_management_bp)
    app.register_blueprint(feedback_bp)

    app.logger.info("Flask application successfully created and configured.")

    return app

# Create app
app = create_app()

# Preload XGBoost for fastest predictions (safest model)
with app.app_context():
    preload_xgboost_on_startup()

if __name__ == '__main__':
    # Start app
    app.run(host='0.0.0.0', port=5100, debug=False)