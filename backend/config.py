import os

class Config:
    # Flask Template & Static
    TEMPLATE_DIR = os.path.abspath('dashboard/templates')
    STATIC_DIR = os.path.abspath('dashboard/static')

    # CORS Einstellungen - Für lokalen Betrieb und ngrok
    CORS_RESOURCES = {
        r"/*": {
            "origins": [
                "https://masterthesis-igbq.onrender.com",
                "http://localhost:3000",
                "http://127.0.0.1:3000"
            ],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "X-User-Uid"],
            "supports_credentials": True
        }
    }

    # Server-Konfiguration für lokalen Betrieb
    HOST = '0.0.0.0'  # Erlaubt Zugriff von allen IPs
    PORT = 5100
    
    # Öffentliche URL (wird später gesetzt)
    PUBLIC_URL = os.environ.get('PUBLIC_URL', 'http://localhost:5100')
    # LinkedIn API (in Produktion über Umgebungsvariablen!)
    LINKEDIN_EMAIL = os.environ.get('LINKEDIN_EMAIL', 'f.runkel@yahoo.com')
    LINKEDIN_PASSWORD = os.environ.get('LINKEDIN_PASSWORD', 'Cool0089!%')

