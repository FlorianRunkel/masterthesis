from flask import Blueprint, request, jsonify, current_app
from linkedin_api import Linkedin
from backend.config import Config

import logging
import json

# Blueprint für LinkedIn-Routen
linkedin_bp = Blueprint('linkedin_bp', __name__)

# Globale Variable für die API-Instanz
linkedin_api = None

def get_linkedin_api_instance():
    """Erstellt oder gibt die bestehende LinkedIn API-Instanz zurück."""
    global linkedin_api
    if linkedin_api is None:
        try:
            email = Config.LINKEDIN_EMAIL
            password = Config.LINKEDIN_PASSWORD
            if not email or not password:
                logging.error("LinkedIn-Anmeldeinformationen sind nicht in der Konfiguration gesetzt.")
                return None
            linkedin_api = Linkedin(email, password, refresh_cookies=True)
            logging.info("LinkedIn API erfolgreich initialisiert")
        except Exception as e:
            logging.error(f"Fehler bei der LinkedIn API Initialisierung: {str(e)}")
            linkedin_api = None # Bei Fehler zurücksetzen
            return None
    return linkedin_api

@linkedin_bp.route('/scrape-linkedin', methods=['POST'])
def scrape_linkedin():
    try:
        data = request.get_json()
        linkedin_url = data.get('url')

        if not linkedin_url:
            return jsonify({'error': 'Keine LinkedIn-URL angegeben'}), 400

        # Extrahiere public identifier aus der URL
        import re
        match = re.search(r"linkedin.com/in/([^/?]+)", linkedin_url)
        if not match:
            return jsonify({'error': 'Ungültige LinkedIn-URL'}), 400
        public_identifier = match.group(1)

        api = get_linkedin_api_instance()
        if not api:
            return jsonify({'error': 'LinkedIn API konnte nicht initialisiert werden.'}), 500

        try:
            profile = api.get_profile(public_identifier)
        except Exception as api_error:
            logging.error(f"LinkedIn API Fehler: {str(api_error)}")
            return jsonify({'error': 'Fehler beim Abrufen der Profildaten von LinkedIn.'}), 500

        # Mapping ins gewünschte Format
        profile_data = {
            'name': profile.get('firstName', '') + ' ' + profile.get('lastName', ''),
            'currentTitle': profile.get('headline', ''),
            'location': profile.get('locationName', ''),
            'imageUrl': profile.get('profilePictureDisplayImage', ''),
            'experience': [],
            'education': [],
            'industry': profile.get('industryName', ''),
            'summary': profile.get('summary', '')
        }
        for exp in profile.get('experience', []):
            profile_data['experience'].append({
                'title': exp.get('title', ''),
                'company': exp.get('companyName', ''),
                'startDate': exp.get('timePeriod', {}).get('startDate', ''),
                'endDate': exp.get('timePeriod', {}).get('endDate', '')
            })
        for edu in profile.get('education', []):
            profile_data['education'].append({
                'degree': edu.get('degreeName', ''),
                'school': edu.get('schoolName', ''),
                'startDate': edu.get('timePeriod', {}).get('startDate', ''),
                'endDate': edu.get('timePeriod', {}).get('endDate', '')
            })
        return jsonify(profile_data)

    except Exception as e:
        logging.error(f"Unerwarteter Fehler beim API-Zugriff: {str(e)}")
        return jsonify({'error': str(e)}), 500 