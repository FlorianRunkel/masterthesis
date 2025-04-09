from flask import Flask, render_template, request, jsonify
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os

# Flask-App initialisieren mit korrektem Template-Verzeichnis
template_dir = os.path.abspath('dashboard/templates')
static_dir = os.path.abspath('dashboard/static')
app = Flask(__name__, 
            template_folder=template_dir,
            static_folder=static_dir)

app.logger.setLevel(logging.INFO)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict_career():
    try:
        data = request.get_json()

        # Extrahiere Nutzereingaben
        first_name = data.get('firstName', '')
        last_name = data.get('lastName', '')
        location = data.get('location', '')
        experiences = data.get('experiences', [])
        model_type = data.get('modelType', 'transformer')

        app.logger.info(f"Eingehende Daten: {data}")

        # Erstelle ein DataFrame mit den Nutzereingaben
        user_data = pd.DataFrame({
            'first_name': [first_name],
            'last_name': [last_name],
            'location': [location],
            'experience_years': [len(experiences)],
            'current_position': [experiences[-1]['position'] if experiences else ''],
            'current_company': [experiences[-1]['company'] if experiences else ''],
            'created_at': [datetime.now()]
        })

        model_predictors = {
            "gru": "ml_pipe.models.gru.predict",
            "xgboost": "ml_pipe.models.xgboost.predict",
            "tft": "ml_pipe.models.tft.predict"
        }

        if model_type not in model_predictors:
            return jsonify({'error': f"Unbekannter Modelltyp: {model_type}"}), 400

        module = __import__(model_predictors[model_type], fromlist=['predict'])
        prediction = module.predict(user_data)

        # Formatiere die Vorhersage
        formatted_prediction = {
            'confidence': prediction['confidence'][0],
            'recommendations': prediction['recommendations'][0]
        }

        return jsonify(formatted_prediction)

    except Exception as e:
        app.logger.error(f"Fehler bei der Vorhersage: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5100, debug=True)