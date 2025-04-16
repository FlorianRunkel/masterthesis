from flask import Flask, render_template, request, jsonify
import pandas as pd
import logging
import os
import json
from linkedin_api import Linkedin

# Flask-App initialisieren mit korrektem Template-Verzeichnis
template_dir = os.path.abspath('dashboard/templates')
static_dir = os.path.abspath('dashboard/static')
app = Flask(__name__, 
            template_folder=template_dir,
            static_folder=static_dir)

app.logger.setLevel(logging.INFO)

# LinkedIn API Konfiguration
LINKEDIN_EMAIL = 'f.runkel@yahoo.com'  # In Produktion über Umgebungsvariablen!
LINKEDIN_PASSWORD = 'Cool0089!%'       # In Produktion über Umgebungsvariablen!

# Globale Variable für die API
linkedin_api = None

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/batch')
def batch():
    return render_template('batch.html')

@app.route('/linkedin')
def linkedin():
    return render_template('linkedin.html')

@app.route('/predict', methods=['POST'])
def predict_career():
    try:
        data = request.get_json()

        experiences = data.get('experiences', [])
        model_type = data.get('modelType', 'tft')

        app.logger.info(f"Eingehende Daten: {data}")

        # Dynamisches Laden des Modell-Moduls
        model_predictors = {
            "gru": "ml_pipe.models.gru.predict",
            "xgboost": "ml_pipe.models.xgboost.predict",
            "tft": "ml_pipe.models.tft.predict"
        }

        if model_type not in model_predictors:
            return jsonify({'error': f"Unbekannter Modelltyp: {model_type}"}), 400

        module = __import__(model_predictors[model_type], fromlist=['predict'])

        # Richtige Input-Struktur für die Modelle
        prediction = module.predict({
            "career_history": experiences
        })

        # Formatiere Vorhersage
        formatted_prediction = {
            'confidence': prediction['confidence'][0],
            'recommendations': prediction['recommendations'][0]
        }

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
            module = __import__("ml_pipe.models.tft.predict", fromlist=['predict'])
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
                
                # Check for required data
                work_experiences = profile_data.get("workExperience", [])
                if not work_experiences:
                    results.append({
                        "firstName": row.get("firstName", ""),
                        "lastName": row.get("lastName", ""),
                        "linkedinProfile": row.get("profileLink", ""),
                        "error": "Keine Berufserfahrung gefunden"
                    })
                    continue
                
                formatted_input = {
                    "career_history": work_experiences
                }
                
                # Make prediction
                prediction = module.predict(formatted_input)
                
                # Check if prediction contains error
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
                        "recommendations": prediction["recommendations"]
                    })

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
            return jsonify({'error': 'Ungültiges LinkedIn-URL-Format'}), 400

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
                'experience': []
            }

            # Formatiere die Berufserfahrung
            for position in profile.get('experience', []):
                exp_data = {
                    'title': position.get('title', ''),
                    'company': position.get('companyName', ''),
                    'duration': f"{position.get('timePeriod', {}).get('startDate', {}).get('year', '')} - "
                              f"{position.get('timePeriod', {}).get('endDate', {}).get('year', 'Present')}"
                }
                profile_data['experience'].append(exp_data)

            # Füge zusätzliche Informationen hinzu
            profile_data['industry'] = profile.get('industry', '')
            profile_data['summary'] = profile.get('summary', '')

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5100, debug=True)