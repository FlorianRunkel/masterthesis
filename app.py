from flask import Flask, render_template, request, jsonify
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
from ml_pipe.data.dummy_data import create_dummy_database
from ml_pipe.predict import predict

# Flask-App initialisieren mit korrektem Template-Verzeichnis
template_dir = os.path.abspath('dashboard/templates')
static_dir = os.path.abspath('dashboard/static')
app = Flask(__name__, 
            template_folder=template_dir,
            static_folder=static_dir)

app.logger.setLevel(logging.INFO)

# Initialisiere die Dummy-Datenbank
create_dummy_database()

def get_db_connection():
    """Erstellt eine Verbindung zur SQLite-Datenbank"""
    conn = sqlite3.connect('ml_pipe/data/database/linkedin_data.db')
    conn.row_factory = sqlite3.Row
    return conn

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
        
        # Führe die Vorhersage durch
        prediction = predict(user_data)
        
        # Formatiere die Vorhersage
        formatted_prediction = {
            'next_career_step': prediction['next_career_step'][0],
            'confidence': prediction['confidence'][0],
            'recommendations': prediction['recommendations'][0]
        }
        
        return jsonify(formatted_prediction)
        
    except Exception as e:
        app.logger.error(f"Fehler bei der Vorhersage: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/profiles', methods=['GET'])
def get_profiles():
    """API-Endpunkt zum Abrufen von Profilen"""
    try:
        conn = get_db_connection()
        profiles = pd.read_sql_query('SELECT * FROM profiles', conn)
        conn.close()
        
        return jsonify(profiles.to_dict(orient='records'))
    except Exception as e:
        app.logger.error(f"Fehler beim Abrufen der Profile: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/experiences/<int:profile_id>', methods=['GET'])
def get_experiences(profile_id):
    """API-Endpunkt zum Abrufen der Erfahrungen eines Profils"""
    try:
        conn = get_db_connection()
        experiences = pd.read_sql_query(
            'SELECT * FROM experiences WHERE profile_id = ?',
            conn,
            params=(profile_id,)
        )
        conn.close()
        
        return jsonify(experiences.to_dict(orient='records'))
    except Exception as e:
        app.logger.error(f"Fehler beim Abrufen der Erfahrungen: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/education/<int:profile_id>', methods=['GET'])
def get_education(profile_id):
    """API-Endpunkt zum Abrufen der Ausbildung eines Profils"""
    try:
        conn = get_db_connection()
        education = pd.read_sql_query(
            'SELECT * FROM education WHERE profile_id = ?',
            conn,
            params=(profile_id,)
        )
        conn.close()
        
        return jsonify(education.to_dict(orient='records'))
    except Exception as e:
        app.logger.error(f"Fehler beim Abrufen der Ausbildung: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)