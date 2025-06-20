from flask import Blueprint, request, jsonify
from backend.ml_pipe.data.database.mongodb import MongoDb
import logging

# Blueprint für Kandidaten-Routen
candidates_bp = Blueprint('candidates_bp', __name__)

# Hilfsfunktion, die hier benötigt wird
def candidate_exists(candidate, mongo_db, collection_name):
    """Prüft, ob ein Kandidat bereits in der Datenbank existiert."""
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

@candidates_bp.route('/candidates', methods=['GET'])
def get_all_candidates():
    """Gibt alle Kandidaten aus der Datenbank zurück."""
    try:
        mongo_db = MongoDb()
        result = mongo_db.get_all('candidates')
        
        if result['statusCode'] != 200:
            return jsonify({'error': result.get('error', 'Unbekannter Fehler')}), result['statusCode']
            
        return jsonify(result['data']), 200
        
    except Exception as e:
        logging.error(f"Fehler beim Abrufen der Kandidaten: {str(e)}")
        return jsonify({'error': str(e)}), 500

@candidates_bp.route('/api/candidates', methods=['POST'])
def save_candidates():
    """Speichert eine Liste von Kandidaten in der Datenbank."""
    try:
        candidates = request.json
        uid = request.headers.get('X-User-Uid')
        
        if not uid:
            if isinstance(candidates, dict) and 'uid' in candidates:
                uid = candidates['uid']
                candidates = candidates.get('candidates', [])
        
        if not uid:
            return jsonify({'error': 'Keine User-UID übergeben!'}), 400
        if not candidates:
            return jsonify({'error': 'Keine Kandidaten zum Speichern gefunden'}), 400
            
        mongo_db = MongoDb()
        if mongo_db.db is None:
            mongo_db.get_mongo_client()
            if mongo_db.db is None:
                return jsonify({'error': 'Fehler bei der Verbindung zur Datenbank'}), 500
                
        saved_count = 0
        skipped_count = 0
        for candidate in candidates:
            logging.info(f"Speichere Kandidaten: {candidate}")
            candidate['uid'] = uid
            if not candidate_exists(candidate, mongo_db, 'candidates'):
                result = mongo_db.create(candidate, 'candidates')
            else:
                result = {'statusCode': 409, 'error': 'Kandidat existiert bereits.'}
                
            if result['statusCode'] == 200: 
                saved_count += 1
                logging.info(f"Kandidat erfolgreich gespeichert: {candidate.get('linkedinProfile','')} (UID: {uid})")
            else:
                skipped_count += 1
                logging.info(f"Fehler beim Speichern des Kandidaten: {result['error']}")
                
        return jsonify({
            'message': 'Kandidaten erfolgreich gespeichert',
            'savedCount': saved_count,
            'skippedCount': skipped_count,
            'reasonSkipped': 'Duplikate basierend auf LinkedIn-Profil-URL'
        }), 201
        
    except Exception as e:
        logging.error(f"Fehler beim Speichern der Kandidaten: {str(e)}")
        return jsonify({'error': 'Interner Serverfehler: ' + str(e)}), 500 