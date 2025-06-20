from flask import Blueprint, request, jsonify, current_app
from linkedin_api import Linkedin
from backend.config import Config
import logging
import json

# Blueprint für LinkedIn-Routen
linkedin_bp = Blueprint('linkedin_bp', __name__)

# Globale Variable für die API-Instanz
linkedin_api = None

def initialize_linkedin_api():
    """Initialisiert die LinkedIn API und speichert sie in der globalen Variable."""
    global linkedin_api
    try:
        # Sicherstellen, dass die Konfiguration geladen ist
        email = Config.LINKEDIN_EMAIL
        password = Config.LINKEDIN_PASSWORD
        if not email or not password:
            logging.error("LinkedIn-Anmeldeinformationen sind nicht in der Konfiguration gesetzt.")
            return False

        linkedin_api = Linkedin(email, password)
        logging.info("LinkedIn API erfolgreich initialisiert")
        
        # Test-Aufruf, um die Verbindung zu verifizieren
        test_profile = linkedin_api.get_profile('williamhgates')
        if test_profile:
            logging.info("LinkedIn API Verbindung erfolgreich getestet")
        return True
    except Exception as e:
        logging.error(f"Fehler bei der LinkedIn API Initialisierung: {str(e)}")
        return False

@linkedin_bp.route('/scrape-linkedin', methods=['POST'])
def scrape_linkedin():
    """Scrapt ein LinkedIn-Profil und gibt die formatierten Daten zurück."""
    global linkedin_api
    
    try:
        if linkedin_api is None:
            if not initialize_linkedin_api():
                return jsonify({'error': 'LinkedIn API nicht verfügbar. Bitte versuchen Sie es später erneut.'}), 500

        data = request.get_json()
        linkedin_url = data.get('url')

        if not linkedin_url:
            return jsonify({'error': 'Keine LinkedIn-URL angegeben'}), 400

        try:
            username = linkedin_url.split('/in/')[1].split('/')[0]
            username = username.split('?')[0]
        except IndexError:
            return jsonify({'error': f"Ungültiges LinkedIn-URL-Format: {linkedin_url}"}), 400

        logging.info(f"Starte API-Abfrage für LinkedIn-Profil: {username}")

        try:
            profile = linkedin_api.get_profile(username)
            if not profile:
                return jsonify({'error': 'Profil nicht gefunden'}), 404

            # Profildaten formatieren
            profile_data = {
                'name': f"{profile.get('firstName', '')} {profile.get('lastName', '')}".strip(),
                'currentTitle': profile.get('headline', ''),
                'location': profile.get('locationName', ''),
                'imageUrl': profile.get('displayPictureUrl', '') + profile.get('img_400_400', ''),
                'experience': [],
                'education': []
            }

            for position in profile.get('experience', []):
                start = position.get('timePeriod', {}).get('startDate', {})
                start_year = str(start.get('year', ''))
                start_month = str(start.get('month', '')).zfill(2) if start.get('month') else '01'
                start_day = str(start.get('day', '')).zfill(2) if start.get('day') else '01'
                start_date = f"{start_day}/{start_month}/{start_year}" if start_year else ''
                end = position.get('timePeriod', {}).get('endDate', {})
                end_date = 'Present'
                if end:
                    end_year = str(end.get('year', ''))
                    end_month = str(end.get('month', '')).zfill(2) if end.get('month') else '01'
                    end_day = str(end.get('day', '')).zfill(2) if end.get('day') else '01'
                    end_date = f"{end_day}/{end_month}/{end_year}" if end_year else 'Present'
                
                profile_data['experience'].append({
                    'title': position.get('title', ''),
                    'company': position.get('companyName', ''),
                    'startDate': start_date,
                    'endDate': end_date
                })

            profile_data['industry'] = profile.get('industry', '')
            profile_data['summary'] = profile.get('summary', '')

            for edu in profile.get('education', []):
                profile_data['education'].append({
                    'degree': edu.get('degreeName', ''),
                    'school': edu.get('schoolName', ''),
                    'startDate': str(edu.get('timePeriod', {}).get('startDate', {}).get('year', '')) if edu.get('timePeriod', {}).get('startDate') else '',
                    'endDate': str(edu.get('timePeriod', {}).get('endDate', {}).get('year', '')) if edu.get('timePeriod', {}).get('endDate') else ''
                })

            logging.info(f"Profildaten erfolgreich extrahiert für: {profile_data['name']}")
            return jsonify(profile_data)

        except Exception as api_error:
            logging.error(f"API-Fehler: {str(api_error)}")
            initialize_linkedin_api()
            return jsonify({'error': 'Fehler beim Abrufen der Profildaten. Bitte versuchen Sie es erneut.'}), 500

    except Exception as e:
        logging.error(f"Unerwarteter Fehler beim API-Zugriff: {str(e)}")
        return jsonify({'error': str(e)}), 500 