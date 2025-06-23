import os
import sys
import logging
from flask import Flask
from flask_cors import CORS

# Füge den Backend-Ordner zum Python-Path hinzu
backend_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(backend_dir)
sys.path.insert(0, parent_dir)

# Lokale Konfiguration und Handler importieren
from backend.config import Config
from backend.api_handlers.pages.handler import pages_bp
from backend.api_handlers.candidates.handler import candidates_bp
from backend.api_handlers.linkedin.handler import linkedin_bp, initialize_linkedin_api
from backend.api_handlers.prediction.handler import prediction_bp
from backend.api_handlers.user_management.handler import user_management_bp

def create_app():
    """Erstellt und konfiguriert die Flask-Anwendung."""
    
    app = Flask(__name__,
                template_folder=Config.TEMPLATE_DIR,
                static_folder=Config.STATIC_DIR)

            # Logging konfigurieren
    app.logger.setLevel(logging.INFO)

    # CORS für alle Routen aktivieren
    CORS(app)

    # Blueprints registrieren, um die Routen zu aktivieren
    app.register_blueprint(pages_bp)
    app.register_blueprint(candidates_bp)
    app.register_blueprint(linkedin_bp)
    app.register_blueprint(prediction_bp)
    app.register_blueprint(user_management_bp)

    # App-Kontext für Initialisierungen verwenden
    with app.app_context():
        # LinkedIn API initialisieren, falls benötigt
        initialize_linkedin_api()
        # Hier könnten weitere Initialisierungen stattfinden (z.B. DB-Check)
    
    app.logger.info("Flask-Anwendung erfolgreich erstellt und konfiguriert.")
    
    return app

# App erstellen
app = create_app()

if __name__ == '__main__':
    # Starte die App
    app.run(host='0.0.0.0', port=5100, debug=True)