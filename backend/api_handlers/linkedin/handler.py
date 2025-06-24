from flask import Blueprint, request, jsonify, current_app
from backend.config import Config
import logging
import json
import requests
import re

# Blueprint für LinkedIn-Routen
linkedin_bp = Blueprint('linkedin_bp', __name__)

API_KEY = "/6Lo4oDo.z/355AyQe6OLY+W9WxnZYe+RGE4YdOHm+WNUHCbt2Zw="
ACCOUNT_ID = "l648QWf-Sw2qMGt499otjA"
DSN = "api16.unipile.com:14659"

def extract_public_identifier(linkedin_url):
    match = re.search(r"linkedin\.com/in/([^/?]+)", linkedin_url)
    if match:
        return match.group(1)
    return None

def fetch_profile_direct(linkedin_url):
    url = f"https://{DSN}/api/v1/linkedin"
    headers = {
        "X-API-KEY": API_KEY,
        "accept": "application/json",
        "content-type": "application/json"
    }
    data = {
        "request_url": linkedin_url,
        "account_id": ACCOUNT_ID
    }
    response = requests.post(url, headers=headers, json=data)
    if response.ok:
        return response.json()
    else:
        print("Fehler:", response.status_code, response.text)
        return None

@linkedin_bp.route('/scrape-linkedin', methods=['POST'])
def scrape_linkedin():
    try:
        data = request.get_json()
        linkedin_url = data.get('url')

        if not linkedin_url:
            return jsonify({'error': 'Keine LinkedIn-URL angegeben'}), 400

        # 1. Profilnamen extrahieren
        public_identifier = extract_public_identifier(linkedin_url)
        if not public_identifier:
            return jsonify({'error': 'Ungültige LinkedIn-URL'}), 400

        # 2. Suche mit Unipile
        try:
            profile = fetch_profile_direct(linkedin_url)
        except Exception as api_error:
            logging.error(f"Unipile Search Fehler: {str(api_error)}")
            return jsonify({'error': 'Fehler beim Abrufen der Profildetails von Unipile.'}), 500
        

        logging.error(f"Unipile Search: {str(profile)}")
        # 5. Mapping ins gewünschte Format
        profile_data = {
            'name': profile.get('full_name', ''),
            'currentTitle': profile.get('headline', ''),
            'location': profile.get('location', ''),
            'imageUrl': profile.get('profile_picture_url', ''),
            'experience': [],
            'education': [],
            'industry': profile.get('industry', ''),
            'summary': profile.get('summary', '')
        }
        for exp in profile.get('experience', []):
            profile_data['experience'].append({
                'title': exp.get('title', ''),
                'company': exp.get('company', ''),
                'startDate': exp.get('start_date', ''),
                'endDate': exp.get('end_date', '')
            })
        for edu in profile.get('education', []):
            profile_data['education'].append({
                'degree': edu.get('degree', ''),
                'school': edu.get('school', ''),
                'startDate': edu.get('start_date', ''),
                'endDate': edu.get('end_date', '')
            })
        return jsonify(profile_data)

    except Exception as e:
        logging.error(f"Unerwarteter Fehler beim API-Zugriff: {str(e)}")
        return jsonify({'error': str(e)}), 500 