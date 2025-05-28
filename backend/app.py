import os
import sys

# Füge den Backend-Ordner zum Python-Path hinzu
backend_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(backend_dir)
sys.path.insert(0, parent_dir)

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import logging
import json
from linkedin_api import Linkedin
from datetime import datetime
from backend.ml_pipe.data.database.mongodb import MongoDb
from backend.config import Config

'''
Flask App - initialisation
'''
# Flask-App initialisieren mit korrektem Template-Verzeichnis
app = Flask(__name__,
            template_folder=Config.TEMPLATE_DIR,
            static_folder=Config.STATIC_DIR)

# CORS für alle Routen aktivieren
CORS(app, resources=Config.CORS_RESOURCES)

app.logger.setLevel(logging.INFO)

# LinkedIn API Konfiguration
LINKEDIN_EMAIL = Config.LINKEDIN_EMAIL
LINKEDIN_PASSWORD = Config.LINKEDIN_PASSWORD

# Globale Variable für die API
linkedin_api = None

# Globale Variable für gespeicherte Kandidaten
candidates_db = []

# MongoDB Instanz erstellen
mongo_db = MongoDb()

'''
Helper functions
'''
def initialize_linkedin_api():
    global linkedin_api
    try:
        linkedin_api = Linkedin(LINKEDIN_EMAIL, LINKEDIN_PASSWORD)
        app.logger.info("LinkedIn API erfolgreich initialisiert")
        # Test-Aufruf um die Verbindung zu verifizieren
        test_profile = linkedin_api.get_profile('williamhgates')
        if test_profile:
            app.logger.info("LinkedIn API Verbindung erfolgreich getestet")
        return True
    except Exception as e:
        app.logger.error(f"Fehler bei der LinkedIn API Initialisierung: {str(e)}")
        return False

def preprocess_dates_time(data):
    import re
    def to_mm_yyyy(date_str):
        if not date_str or date_str == "Present":
            return "Present"
        if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            year, month, _ = date_str.split('-')
            return f"{month}/{year}"
        if re.match(r"^\d{4}$", date_str):
            return f"01/{date_str}"
        if re.match(r"^\d{2}/\d{4}$", date_str):
            return date_str
        return date_str

    # Für verschiedene mögliche Feldnamen
    for exp_key in ['experience', 'workExperience']:
        for exp in data.get(exp_key, []):
            exp['startDate'] = to_mm_yyyy(exp.get('startDate', ''))
            exp['endDate'] = to_mm_yyyy(exp.get('endDate', ''))
    for edu in data.get('education', []):
        edu['startDate'] = to_mm_yyyy(edu.get('startDate', ''))
        edu['endDate'] = to_mm_yyyy(edu.get('endDate', ''))
    return data

def candidate_exists(candidate, mongo_db, collection_name):
    # Prüfe auf LinkedIn-Profil
    if candidate.get('linkedinProfile'):
        res = mongo_db.get({'linkedinProfile': candidate['linkedinProfile']}, collection_name)
        if res['statusCode'] == 200 and res['data']:
            return True
    # Prüfe auf Vor- und Nachname (nur wenn LinkedIn nicht vorhanden oder leer)
    if candidate.get('firstName') and candidate.get('lastName'):
        res = mongo_db.get({'firstName': candidate['firstName'], 'lastName': candidate['lastName']}, collection_name)
        if res['statusCode'] == 200 and res['data']:
            return True
    return False

'''
Routes
'''
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/batch')
def batch():
    return render_template('batch.html')

@app.route('/linkedin')
def linkedin():
    return render_template('linkedin.html')

'''
functions for pages
'''
@app.route('/candidates')
def get_all_candidates():
    try:
        result = mongo_db.get_all('candidates')
        
        if result['statusCode'] != 200:
            return jsonify({'error': result.get('error', 'Unbekannter Fehler')}), result['statusCode']
            
        return jsonify(result['data']), 200
        
    except Exception as e:
        app.logger.error(f"Fehler beim Abrufen der Kandidaten: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_career():
    try:
        data = request.get_json()
        app.logger.info(f"Eingehende Daten: {data}")

        # Extrahiere die Profilinformationen
        if "linkedinProfileInformation" in data:
            try:
                profile_data = json.loads(data["linkedinProfileInformation"])
            except Exception as json_err:
                app.logger.error(f"JSON parsing error: {str(json_err)}")
                return jsonify({'error': f'Ungültiges JSON-Format: {str(json_err)}'}), 400
        else:
            profile_data = data

        model_type = data.get('modelType', 'tft').lower()
        app.logger.info(f"Verwende Modell: {model_type}")

        # Dynamisches Laden des Modell-Moduls
        model_predictors = {
            "gru": "backend.ml_pipe.models.gru.predict",
            "xgboost": "backend.ml_pipe.models.xgboost.predict",
            "tft": "backend.ml_pipe.models.tft.predict"
        }

        if model_type not in model_predictors:
            app.logger.error(f"Unbekannter Modelltyp: {model_type}")
            return jsonify({'error': f"Unbekannter Modelltyp: {model_type}"}), 400

        module = __import__(model_predictors[model_type], fromlist=['predict'])
        app.logger.info(f"Modul erfolgreich geladen: {model_predictors[model_type]}")

        # Vorhersage mit den Profildaten
        if model_type == 'gru' or model_type == 'tft':
            profile_data = preprocess_dates_time(profile_data)

        prediction = module.predict(profile_data, with_llm_explanation=True)
    
        # Formatiere Vorhersage
        if model_type == 'xgboost':
            formatted_prediction = {
                'confidence': max(0.0, prediction['confidence'][0]),
                'recommendations': prediction['recommendations'][0],
                'status': prediction.get('status', ''),
                'explanations': prediction.get('explanations', []),
            }
        else:
            formatted_prediction = {
                'confidence': prediction['confidence'],
                'recommendations': prediction['recommendations'],
                'status': prediction.get('status', ''),
                'explanations': prediction.get('explanations', []),
            }
        
        app.logger.info(f"Formatted Prediction: {formatted_prediction}")

        return jsonify(formatted_prediction)

    except Exception as e:
        app.logger.error(f"Fehler bei der Vorhersage: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    try:
        # Datei aus dem Request holen
        if 'file' not in request.files:
            return jsonify({'error': 'Keine Datei im Request gefunden'}), 400

        file = request.files['file']
        app.logger.info(f"Received file: {file.filename}")
        
        # Try to read the CSV
        try:
            df = pd.read_csv(file)
            app.logger.info(f"CSV loaded successfully with {len(df)} rows")
        except Exception as csv_err:
            app.logger.error(f"CSV parsing error: {str(csv_err)}")
            return jsonify({'error': f'Fehler beim Lesen der CSV-Datei: {str(csv_err)}'}), 400

        # Import model dynamically
        try:
            model_type = request.form.get('modelType', 'xgboost').lower()
            model_predictors = {
                "gru": "backend.ml_pipe.models.gru.predict",
                "xgboost": "backend.ml_pipe.models.xgboost.predict",
                "tft": "backend.ml_pipe.models.tft.predict"
            }

            if model_type not in model_predictors:
                return jsonify({'error': f"Unbekannter Modelltyp: {model_type}"}), 400

            module = __import__(model_predictors[model_type], fromlist=['predict'])
            app.logger.info("Model module imported successfully")
        except Exception as import_err:
            app.logger.error(f"Model import error: {str(import_err)}")
            return jsonify({'error': f'Fehler beim Laden des Modells: {str(import_err)}'}), 500

        results = []

        for idx, row in df.iterrows():
            try:
                app.logger.info(f"Processing row {idx+1}/{len(df)}")
                
                # Make sure we handle possible missing data
                if "linkedinProfileInformation" not in row or pd.isna(row["linkedinProfileInformation"]):
                    results.append({
                        "firstName": row.get("firstName", ""),
                        "lastName": row.get("lastName", ""),
                        "linkedinProfile": row.get("profileLink", ""),
                        "error": "Fehlende LinkedIn-Profilinformationen"
                    })
                    continue
                
                # Extract profile data
                try:
                    profile_data = json.loads(row["linkedinProfileInformation"])
                except Exception as json_err:
                    app.logger.error(f"JSON parsing error for row {idx+1}: {str(json_err)}")
                    results.append({
                        "firstName": row.get("firstName", ""),
                        "lastName": row.get("lastName", ""),
                        "linkedinProfile": row.get("profileLink", ""),
                        "error": f"Ungültiges JSON-Format: {str(json_err)}"
                    })
                    continue
                    
                # Make prediction with complete profile data
                if model_type == 'tft':
                    profile_data = preprocess_dates_time(profile_data)

                app.logger.info(f"Profile data: {profile_data}")
                prediction = module.predict(profile_data, with_llm_explanation=False)
                app.logger.info(f"prediction: {prediction}")

                if "error" in prediction:
                    results.append({
                        "firstName": row.get("firstName", ""),
                        "lastName": row.get("lastName", ""),
                        "linkedinProfile": row.get("profileLink", ""),
                        "error": prediction["error"]
                    })
                else:
                    results.append({
                        "firstName": row.get("firstName", ""),
                        "lastName": row.get("lastName", ""),
                        "linkedinProfile": row.get("profileLink", ""),
                        "confidence": prediction["confidence"],
                        "recommendations": prediction["recommendations"],
                        "status": prediction.get("status", ""),
                        "explanations": prediction.get("explanations", []),
                    })
                    print(results)

            except Exception as user_err:
                app.logger.error(f"Error processing row {idx+1}: {str(user_err)}")
                results.append({
                    "firstName": row.get("firstName", ""),
                    "lastName": row.get("lastName", ""),
                    "linkedinProfile": row.get("profileLink", ""),
                    "error": f"Fehler bei der Verarbeitung: {str(user_err)}"
                })

        app.logger.info(f"Processing completed. Successful: {sum(1 for r in results if 'error' not in r)}, Failed: {sum(1 for r in results if 'error' in r)}")
        return jsonify({"results": results})

    except Exception as e:
        app.logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/scrape-linkedin', methods=['POST'])
def scrape_linkedin():
    global linkedin_api
    
    try:
        # Überprüfe ob die API initialisiert ist
        if linkedin_api is None:
            if not initialize_linkedin_api():
                return jsonify({
                    'error': 'LinkedIn API nicht verfügbar. Bitte versuchen Sie es später erneut.'
                }), 500

        data = request.get_json()
        linkedin_url = data.get('url')

        if not linkedin_url:
            return jsonify({'error': 'Keine LinkedIn-URL angegeben'}), 400

        # Extrahiere LinkedIn Username aus der URL
        try:
            username = linkedin_url.split('/in/')[1].split('/')[0]
            username = username.split('?')[0]  # Entferne Query-Parameter
        except IndexError:
            print(f"Ungültiges LinkedIn-URL-Format: {username}")

        app.logger.info(f"Starte API-Abfrage für LinkedIn-Profil: {username}")

        try:
            # Hole Profildaten über die API
            profile = linkedin_api.get_profile(username)
            #app.logger.info(f"Rohe Profildaten: {json.dumps(profile, indent=2)}")
            
            contact_info = linkedin_api.get_profile_contact_info(username)
            #app.logger.info(f"Kontaktinformationen: {json.dumps(contact_info, indent=2)}")

            if not profile:
                return jsonify({'error': 'Profil nicht gefunden'}), 404

            # Formatiere die Profildaten
            profile_data = {
                'name': f"{profile.get('firstName', '')} {profile.get('lastName', '')}".strip(),
                'currentTitle': profile.get('headline', ''),
                'location': profile.get('locationName', ''),
                'imageUrl': profile.get('displayPictureUrl', '') + profile.get('img_400_400', ''),
                'experience': [],
                'education': []
            }

            # Formatiere die Berufserfahrung
            for position in profile.get('experience', []):
                # Startdatum
                start = position.get('timePeriod', {}).get('startDate', {})
                start_year = str(start.get('year', ''))
                start_month = str(start.get('month', '')).zfill(2) if start.get('month') else '01'
                start_day = str(start.get('day', '')).zfill(2) if start.get('day') else '01'
                start_date = f"{start_day}/{start_month}/{start_year}" if start_year else ''
                # Enddatum
                end = position.get('timePeriod', {}).get('endDate', {})
                if end:
                    end_year = str(end.get('year', ''))
                    end_month = str(end.get('month', '')).zfill(2) if end.get('month') else '01'
                    end_day = str(end.get('day', '')).zfill(2) if end.get('day') else '01'
                    end_date = f"{end_day}/{end_month}/{end_year}" if end_year else ''
                else:
                    end_date = 'Present'
                exp_data = {
                    'title': position.get('title', ''),
                    'company': position.get('companyName', ''),
                    'startDate': start_date,
                    'endDate': end_date
                }
                profile_data['experience'].append(exp_data)

            # Füge zusätzliche Informationen hinzu
            profile_data['industry'] = profile.get('industry', '')
            profile_data['summary'] = profile.get('summary', '')

            # Ausbildung extrahieren
            education_list = []
            for edu in profile.get('education', []):
                edu_data = {
                    'degree': edu.get('degreeName', ''),
                    'school': edu.get('schoolName', ''),
                    'startDate': str(edu.get('timePeriod', {}).get('startDate', {}).get('year', '')) if edu.get('timePeriod', {}).get('startDate') else '',
                    'endDate': str(edu.get('timePeriod', {}).get('endDate', {}).get('year', '')) if edu.get('timePeriod', {}).get('endDate') else ''
                }
                education_list.append(edu_data)
            profile_data['education'] = education_list

            app.logger.info(f"Profildaten erfolgreich extrahiert für: {profile_data['name']}")
            return jsonify(profile_data)

        except Exception as api_error:
            app.logger.error(f"API-Fehler: {str(api_error)}")
            # Versuche die API neu zu initialisieren
            initialize_linkedin_api()
            return jsonify({'error': 'Fehler beim Abrufen der Profildaten. Bitte versuchen Sie es erneut.'}), 500

    except Exception as e:
        app.logger.error(f"Unerwarteter Fehler beim API-Zugriff: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/candidates', methods=['POST'])
def save_candidates():
    try:
        candidates = request.json
        
        if not candidates:
            return jsonify({'error': 'Keine Kandidaten zum Speichern gefunden'}), 400
            
        # Stelle sicher, dass die MongoDB-Verbindung hergestellt ist
        if mongo_db.db is None:
            mongo_db.get_mongo_client()
            if mongo_db.db is None:
                return jsonify({'error': 'Fehler bei der Verbindung zur Datenbank'}), 500
                
        saved_count = 0
        skipped_count = 0
        for candidate in candidates:
            app.logger.info(f"Speichere Kandidaten: {candidate}")

            if not candidate_exists(candidate, mongo_db, 'candidates'):
                result = mongo_db.create(candidate, 'candidates')
            else:
                result = {'statusCode': 409, 'error': 'Kandidat existiert bereits.'}

            if result['statusCode'] == 200: 
                saved_count += 1
                app.logger.info(f"Kandidat erfolgreich gespeichert: {candidate['linkedinProfile']}")
            else:
                skipped_count += 1
                app.logger.info(f"Fehler beim Speichern des Kandidaten: {result['error']}")

        return jsonify({
            'message': 'Kandidaten erfolgreich gespeichert',
            'savedCount': saved_count,
            'skippedCount': skipped_count,
            'reasonSkipped': 'Duplikate basierend auf LinkedIn-Profil-URL'
        }), 201
        
    except Exception as e:
        logging.error(f"Fehler beim Speichern der Kandidaten: {str(e)}")
        return jsonify({'error': 'Interner Serverfehler: ' + str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5100, debug=True)