from flask import Blueprint, request, jsonify
from backend.ml_pipe.data.database.mongodb import MongoDb
import logging

user_management_bp = Blueprint('user_management_bp', __name__)

def create_user_api(first_name, last_name, email, password, canViewExplanations=False):
    """Erstellt einen neuen Benutzer in der Datenbank."""
    mongo_db = MongoDb()
    if not all([first_name, last_name, email, password]):
        return {'statusCode': 400, 'error': 'Alle Felder sind erforderlich'}
    
    # Check, ob User bereits existiert
    if mongo_db.get({'email': email}, 'users')['data']:
        return {'statusCode': 409, 'error': 'Ein Benutzer mit dieser E-Mail existiert bereits'}
        
    # Verwende die create_user-Methode mit dem übergebenen Wert
    return mongo_db.create_user(first_name, last_name, email, password, canViewExplanations)

def get_all_users_api():
    """Ruft alle Benutzer aus der Datenbank ab."""
    mongo_db = MongoDb()
    return mongo_db.get_all('users')

def update_user_api(user_id, update_data):
    """
    Aktualisiert einen Benutzer mit den übergebenen Daten.
    Nimmt ein Dictionary `update_data` entgegen und aktualisiert nur die darin enthaltenen Felder.
    """
    mongo_db = MongoDb()
    if not update_data:
        return {'statusCode': 400, 'error': 'No update data provided.'}
        
    # Erlaubte Felder definieren
    allowed_fields = ['firstName', 'lastName', 'email', 'password', 'canViewExplanations']
    update_dict = {k: v for k, v in update_data.items() if k in allowed_fields}

    # Leeres Passwort nicht speichern
    if 'password' in update_dict and not update_dict['password']:
        del update_dict['password']

    if not update_dict:
        return {'statusCode': 400, 'error': 'No valid fields to update.'}
        
    return mongo_db.update_by_id(user_id, update_dict, 'users')

def delete_user_api(user_id):
    """Löscht einen Benutzer aus der Datenbank."""
    mongo_db = MongoDb()
    return mongo_db.delete_by_id(user_id, 'users')



@user_management_bp.route('/api/login', methods=['POST'])
def login_user():
    """Behandelt den User-Login."""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        if not all([email, password]):
            return jsonify({'error': 'E-Mail und Passwort sind erforderlich.'}), 400
            
        mongo_db = MongoDb()
        result = mongo_db.check_user_credentials(email, password)
        
        if result['statusCode'] == 200:
            user = result['data']
            return jsonify({'message': 'Login successful.', 'user': user}), 200
        else:
            return jsonify({'error': result.get('error', 'Falsche Anmeldedaten')}), 401
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@user_management_bp.route('/api/create-user', methods=['POST'])
def api_create_user():
    """API-Endpunkt zum Erstellen eines neuen Benutzers."""
    try:
        data = request.get_json()
        result = create_user_api(
            first_name=data.get('firstName'),
            last_name=data.get('lastName'),
            email=data.get('email'),
            password=data.get('password'),
            canViewExplanations=data.get('canViewExplanations', False)
        )
        if result['statusCode'] == 200:
            return jsonify({'message': 'User erfolgreich erstellt!', 'data': result['data']}), 200
        else:
            return jsonify({'error': result.get('error', 'Unbekannter Fehler')}), result.get('statusCode', 400)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@user_management_bp.route('/api/users', methods=['GET'])
def api_get_all_users():
    """API-Endpunkt zum Abrufen aller Benutzer."""
    result = get_all_users_api()
    if result['statusCode'] == 200:
        # Die Daten direkt zurückgeben, wie vom Frontend erwartet
        return jsonify(result.get('data', [])), 200
    else:
        return jsonify({'error': result.get('error', 'Unbekannter Fehler')}), result.get('statusCode', 400)

@user_management_bp.route('/api/users/<user_id>', methods=['DELETE'])
def api_delete_user(user_id):
    """API-Endpunkt zum Löschen eines Benutzers."""
    result = delete_user_api(user_id)
    if result['statusCode'] == 200:
        return jsonify({'message': 'User erfolgreich gelöscht'}), 200
    else:
        return jsonify({'error': result.get('error', 'Unbekannter Fehler')}), result.get('statusCode', 400)

@user_management_bp.route('/api/users/<user_id>', methods=['PUT'])
def api_update_user(user_id):
    """API-Endpunkt zum Aktualisieren eines Benutzers."""
    data = request.get_json()
    # Ruft die neue, flexible update_user_api auf
    result = update_user_api(user_id, data)
    
    if result.get('statusCode') == 200:
        return jsonify({'message': 'User erfolgreich aktualisiert'}), 200
    else:
        return jsonify({'error': result.get('error', 'Update fehlgeschlagen')}), result.get('statusCode', 400) 