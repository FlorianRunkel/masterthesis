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

# Global model cache for faster predictions
loaded_models = {}

def load_models_on_startup():
    """Load all ML models at startup to avoid reloading on each prediction"""
    global loaded_models
    try:
        app.logger.info("Loading ML models at startup...")

        # Import prediction modules
        from ml_pipe.models.tft.predict import predict as load_tft_model
        from ml_pipe.models.gru.predict import load_model as load_gru_model
        from ml_pipe.models.xgboost.predict import load_xgb_model

        # Load models into memory
        # For TFT, we need to load the actual model, not the predict function
        from pytorch_forecasting import TemporalFusionTransformer
        tft_model_path = "ml_pipe/models/tft/saved_models/tft_optimized_20250808_122135.ckpt"
        loaded_models['tft'] = TemporalFusionTransformer.load_from_checkpoint(tft_model_path)

        # For GRU and XGBoost, use the load functions
        gru_model_path = "ml_pipe/models/gru/saved_models/gru_model_20250807_184702.pt"
        loaded_models['gru'] = load_gru_model(gru_model_path)

        xgb_model_path = "ml_pipe/models/xgboost/saved_models/xgboost_model_20250806_151510.joblib"
        loaded_models['xgboost'] = load_xgb_model(xgb_model_path)

        app.logger.info("All ML models loaded successfully!")

    except Exception as e:
        app.logger.error(f"Error loading models at startup: {str(e)}")
        # Continue without preloaded models

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

# Load models after app creation
with app.app_context():
    load_models_on_startup()

if __name__ == '__main__':
    # Start app
    app.run(host='0.0.0.0', port=5100, debug=False)