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

        experiences = data.get('experiences', [])
        model_type = data.get('modelType', 'gru')

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5100, debug=True)