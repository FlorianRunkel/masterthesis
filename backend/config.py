import os

class Config:
    # Flask Template & Static
    TEMPLATE_DIR = os.path.abspath('dashboard/templates')
    STATIC_DIR = os.path.abspath('dashboard/static')

    # CORS Einstellungen
    CORS_RESOURCES = {
        r"/*": {
            "origins": [
                "http://localhost:3000",
                "http://localhost:5173",
                "http://192.168.3.10:3000"
            ],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "supports_credentials": True
        }
    }

    # LinkedIn API (in Produktion über Umgebungsvariablen!)
    LINKEDIN_EMAIL = os.environ.get('LINKEDIN_EMAIL', 'f.runkel@yahoo.com')
    LINKEDIN_PASSWORD = os.environ.get('LINKEDIN_PASSWORD', 'Cool0089!%')
