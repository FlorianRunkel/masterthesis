from flask import Blueprint, request, jsonify
from backend.ml_pipe.data.database.mongodb import MongoDb
import logging

# Blueprint für User-Management-Routen
user_management_bp = Blueprint('user_management_bp', __name__)

# --- Interne API-Logik (ehemals admin_api.py) ---

def create_user_api(first_name, last_name, email, password):
    """Erstellt einen neuen Benutzer in der Datenbank."""
    mongo_db = MongoDb()
    if not all([first_name, last_name, email, password]):
        return {'statusCode': 400, 'error': 'Alle Felder sind erforderlich'}
    
    # Check, ob User bereits existiert
    if mongo_db.get({'email': email}, 'users')['data']:
        return {'statusCode': 409, 'error': 'Ein Benutzer mit dieser E-Mail existiert bereits'}
        
    # Verwende die create_user-Methode, die automatisch eine UID vergibt
    return mongo_db.create_user(first_name, last_name, email, password)

def get_all_users_api():
    """Ruft alle Benutzer aus der Datenbank ab."""
    mongo_db = MongoDb()
    return mongo_db.get_all('users')

def update_user_api(user_id, first_name, last_name, email, password):
    """Aktualisiert einen Benutzer in der Datenbank."""
    mongo_db = MongoDb()
    
    update_data = {}
    if first_name is not None:
        update_data['firstName'] = first_name
    if last_name is not None:
        update_data['lastName'] = last_name
    if email is not None:
        update_data['email'] = email
    if password:  # Passwort nur aktualisieren, wenn ein neues angegeben wird
        update_data['password'] = password
        
    if not update_data:
        return {'statusCode': 400, 'error': 'Keine Daten zum Aktualisieren angegeben'}
        
    return mongo_db.update_by_id(user_id, update_data, 'users')

def delete_user_api(user_id):
    """Löscht einen Benutzer aus der Datenbank."""
    mongo_db = MongoDb()
    return mongo_db.delete_by_id(user_id, 'users')


# --- API-Endpunkte / Routen ---

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
            password=data.get('password')
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
    result = update_user_api(
        user_id,
        first_name=data.get('firstName'),
        last_name=data.get('lastName'),
        email=data.get('email'),
        password=data.get('password')
    )
    if result['statusCode'] == 200:
        return jsonify({'message': 'User erfolgreich aktualisiert'}), 200
    else:
        return jsonify({'error': result.get('error', 'Unbekannter Fehler')}), result.get('statusCode', 400) 