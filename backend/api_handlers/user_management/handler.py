from flask import Blueprint, request, jsonify
from backend.ml_pipe.data.database.mongodb import MongoDb
import logging

logger = logging.getLogger(__name__)
user_management_bp = Blueprint('user_management_bp', __name__)

'''
Helper functions
'''
def create_user_helper(first_name, last_name, email, password, canViewExplanations=False):
    try:
        mongo_db = MongoDb()
        if not all([first_name, last_name, email, password]):
            return {'statusCode': 400, 'error': 'All fields are required'}

        if mongo_db.get({'email': email}, 'users')['data']:
            return {'statusCode': 409, 'error': 'U ser with this email already exists'}

        return mongo_db.create_user(first_name, last_name, email, password, canViewExplanations)
    except Exception as e:
        logger.error(f"Error in update_user_api: {e}")
        return {'statusCode': 500, 'error': str(e)}

def get_all_users_helper():
    try:
        mongo_db = MongoDb()
        return mongo_db.get_all('users')
    except Exception as e:
        logger.error(f"Error in get_all_users_api: {e}")
        return {'statusCode': 500, 'error': str(e)}

def update_user_helper(user_id, update_data):
    try:
        mongo_db = MongoDb()
        if not update_data:
            return {'statusCode': 400, 'error': 'No update data provided.'}

        allowed_fields = ['firstName', 'lastName', 'email', 'password', 'canViewExplanations']
        update_dict = {k: v for k, v in update_data.items() if k in allowed_fields}

        if 'password' in update_dict and not update_dict['password']:
            del update_dict['password']

        if not update_dict:
            logger.error(f"No valid fields to update.")
            return {'statusCode': 400, 'error': 'No valid fields to update.'}

        return mongo_db.update_by_id(user_id, update_dict, 'users')
    except Exception as e:
        logger.error(f"Error in update_user_api: {e}")
        return {'statusCode': 500, 'error': str(e)}

def delete_user_helper(user_id):
    try:
        mongo_db = MongoDb()
        return mongo_db.delete_by_id(user_id, 'users')
    except Exception as e:
        logger.error(f"Error in delete_user_api: {e}")
        return {'statusCode': 500, 'error': str(e)}

'''
Login user
'''
@user_management_bp.route('/api/login', methods=['POST'])
def login_user():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        if not all([email, password]):
            return jsonify({'error': 'Email and password are required.'}), 400

        mongo_db = MongoDb()
        result = mongo_db.check_user_credentials(email, password)

        if result['statusCode'] == 200:
            user = result['data']
            return jsonify({'message': 'Login successful.', 'user': user}), 200
        else:
            return jsonify({'error': result.get('error', 'Invalid credentials')}), 401
    except Exception as e:
        logger.error(f"Error in login_user: {e}")
        return jsonify({'error': str(e)}), 500

'''
Create new user
'''
@user_management_bp.route('/api/create-user', methods=['POST'])
def api_create_user():
    try:
        data = request.get_json()
        result = create_user_helper(
            first_name=data.get('firstName'),
            last_name=data.get('lastName'),
            email=data.get('email'),
            password=data.get('password'),
            canViewExplanations=data.get('canViewExplanations', False)
        )
        if result['statusCode'] == 200:
            return jsonify({'message': 'User created successfully', 'data': result['data']}), 200
        else:
            return jsonify({'error': result.get('error', 'User creation failed')}), result.get('statusCode', 400)
    except Exception as e:
        logger.error(f"Error in api_create_user: {e}")
        return jsonify({'error': str(e)}), 500

'''
Get all users
'''
@user_management_bp.route('/api/users', methods=['GET'])
def api_get_all_users():
    try:
        result = get_all_users_helper()
        if result['statusCode'] == 200:
            return jsonify(result.get('data', [])), 200
        else:
            return jsonify({'error': result.get('error', 'Failed to get all users')}), result.get('statusCode', 400)
    except Exception as e:
        logger.error(f"Error in api_get_all_users: {e}")
        return jsonify({'error': str(e)}), 500

'''
Delete user
'''
@user_management_bp.route('/api/users/<user_id>', methods=['DELETE'])
def api_delete_user(user_id):
    try:
        result = delete_user_helper(user_id)
        if result['statusCode'] == 200:
            return jsonify({'message': 'User deleted successfully'}), 200
        else:
            return jsonify({'error': result.get('error', 'Failed to delete user')}), result.get('statusCode', 400)
    except Exception as e:
        logger.error(f"Error in api_delete_user: {e}")
        return jsonify({'error': str(e)}), 500

'''
Update user
'''
@user_management_bp.route('/api/users/<user_id>', methods=['PUT'])
def api_update_user(user_id):
    try:
        data = request.get_json()
        result = update_user_helper(user_id, data)

        if result.get('statusCode') == 200:
            return jsonify({'message': 'User erfolgreich aktualisiert'}), 200
        else:
            return jsonify({'error': result.get('error', 'Update fehlgeschlagen')}), result.get('statusCode', 400)
    except Exception as e:
        logger.error(f"Error in api_update_user: {e}")
        return jsonify({'error': str(e)}), 500