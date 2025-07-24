import logging
from ml_pipe.data.database.mongodb import MongoDb
from bson import ObjectId

logger = logging.getLogger(__name__)

"""
Create a new user in the database.
All fields are required.
"""
def create_user_api(first_name, last_name, email, password, canViewExplanations):
    try:
        if not all([first_name, last_name, email, password, canViewExplanations]):
            return {'statusCode': 400, 'error': 'All fields are required!'}
        # Hinweis: In Produktion Passwort hashen!
        db = MongoDb()
        result = db.create_user(first_name, last_name, email, password, canViewExplanations)
        return result
    except Exception as e:
        logger.error(f"Error in create_user_api: {e}")
        return {'statusCode': 500, 'error': str(e)}

"""
Update an existing user by id.
All fields except user_id are required.
"""
def update_user_api(user_id, first_name, last_name, email, password, canViewExplanations):
    try:
        if not all([first_name, last_name, email, password]):
            return {'statusCode': 400, 'error': 'All fields are required!'}
        db = MongoDb()
        filter_dict = {'_id': ObjectId(user_id)}
        update_dict = {
            'firstName': first_name,
            'lastName': last_name,
            'email': email,
            'password': password,  # Hinweis: Passwort sollte gehasht werden!
            'canViewExplanations': canViewExplanations
        }
        result = db.update(filter_dict, update_dict, 'users')
        return result
    except Exception as e:
        logger.error(f"Error in update_user_api: {e}")
        return {'statusCode': 500, 'error': str(e)}

"""
Delete a user by id.
"""
def delete_user_api(user_id):
    try:
        db = MongoDb()
        result = db.delete_by_id(user_id, 'users')
        return result
    except Exception as e:
        logger.error(f"Error in delete_user_api: {e}")
        return {'statusCode': 500, 'error': str(e)}

"""
Get all users from the database.
"""
def get_all_users_api():
    try:
        db = MongoDb()
        result = db.get_all("users")
        return result
    except Exception as e:
        logger.error(f"Error in get_all_users_api: {e}")
        return {'statusCode': 500, 'error': str(e)} 