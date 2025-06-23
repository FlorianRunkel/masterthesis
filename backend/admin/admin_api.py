import sys
sys.path.append('/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/')

from ml_pipe.data.database.mongodb import MongoDb
from bson import ObjectId

def create_user_api(first_name, last_name, email, password, canViewExplanations):
    if not all([first_name, last_name, email, password, canViewExplanations]):
        return {'statusCode': 400, 'error': 'All fields are required!'}
    db = MongoDb()
    result = db.create_user(first_name, last_name, email, password, canViewExplanations)
    return result 

def update_user_api(user_id, first_name, last_name, email, password, canViewExplanations):
    if not all([first_name, last_name, email, password]):
        return {'statusCode': 400, 'error': 'All fields are required!'}
    db = MongoDb()
    filter_dict = {'_id': ObjectId(user_id)}
    update_dict = {'firstName': first_name, 'lastName': last_name, 'email': email, 'password': password, 'canViewExplanations': canViewExplanations}
    result = db.update(filter_dict, update_dict, 'users')
    return result 

def delete_user_api(user_id):
    db = MongoDb()
    result = db.delete_by_id(user_id, 'users')
    return result 

def get_all_users_api():
    db = MongoDb()
    result = db.get_all("users")
    return result 