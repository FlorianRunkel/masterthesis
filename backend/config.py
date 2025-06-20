import os

class Config:
    # Flask Template & Static
    TEMPLATE_DIR = os.path.abspath('dashboard/templates')
    STATIC_DIR = os.path.abspath('dashboard/static')

    # CORS Einstellungen - Erweitert für öffentlichen Zugriff
    CORS_RESOURCES = {
        r"/*": {
            "origins": "*",  # Erlaubt Anfragen von allen Domains
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "X-User-Uid"],
            "supports_credentials": False
        }
    }

    # Server-Konfiguration für öffentlichen Zugriff
    HOST = '0.0.0.0'  # Erlaubt Zugriff von allen IPs
    PORT = 5100
    
    # Öffentliche URL (wird später gesetzt)
    PUBLIC_URL = os.environ.get('PUBLIC_URL', 'http://localhost:5100')

    # LinkedIn API (in Produktion über Umgebungsvariablen!)
    LINKEDIN_EMAIL = os.environ.get('LINKEDIN_EMAIL', 'f.runkel@yahoo.com')
    LINKEDIN_PASSWORD = os.environ.get('LINKEDIN_PASSWORD', 'Cool0089!%')
