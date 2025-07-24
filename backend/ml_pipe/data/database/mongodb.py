import pymongo
from bson import ObjectId
from pymongo.errors import ConfigurationError, ConnectionFailure, ServerSelectionTimeoutError
import logging
import certifi

# Initialize logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class MongoDb:
    '''
    Initialize & connect to MongoDB
    '''
    def __init__(self, user='florianrunkel', password='ur04mathesis', db_name='Database', url='cluster0.1lgrrsw.mongodb.net'):
        self.user = user
        self.password = password
        self.db_name = db_name
        self.url = url
        self.client = None
        self.db = None

    '''
    Get MongoDB client
    '''
    def get_mongo_client(self):
        if self.client is None:
            try:
                mongo_uri = f"mongodb+srv://{self.user}:{self.password}@{self.url}/{self.db_name}?retryWrites=true&w=majority"
                self.client = pymongo.MongoClient(
                    mongo_uri,
                    tls=True,
                    tlsCAFile=certifi.where(),
                    serverSelectionTimeoutMS=5000
                )
                self.db = self.client[self.db_name]
                self.client.admin.command('ping')
            except ConfigurationError as e:
                logger.error(f"MongoDB configuration error: {e}")
                raise
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                logger.error(f"Connection to MongoDB failed: {e}")
                raise
        return self.db

    '''
    Get collection
    '''
    def get_collection(self, collection_name):
        try:
            db = self.get_mongo_client()
            return db[collection_name]
        except Exception as e:
            logger.error(f"Error getting collection: {e}")
            return None

    '''
    Add document to collection
    '''
    def create(self, document, collection_name):
        try:
            collection = self.get_collection(collection_name)
            result = collection.insert_one(document)
            if result.acknowledged:
                new_document = collection.find_one({'_id': result.inserted_id})
                if new_document and '_id' in new_document:
                    new_document['_id'] = str(new_document['_id'])
                return {'statusCode': 200, 'data': new_document}
            else:
                return {'statusCode': 400, 'error': 'Document could not be added'}
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return {'statusCode': 500, 'error': str(e)}

    '''
    Get document by filter
    '''
    def get(self, filter_dict, collection_name):
        try:
            collection = self.get_collection(collection_name)
            documents = list(collection.find(filter_dict))
            for doc in documents:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
            return {'statusCode': 200, 'data': documents}
        except Exception as e:
            logger.error(f"Fehler beim Suchen: {e}")
            return {'statusCode': 500, 'error': str(e)}

    '''
    Get all documents from collection
    '''
    def get_all(self, collection_name):
        try:
            return self.get({}, collection_name)
        except Exception as e:
            logger.error(f"Error getting all documents: {e}")
            return {'statusCode': 500, 'error': str(e)}

    '''
    Get document by id
    '''
    def get_by_id(self, id, collection_name):
        try:
            collection = self.get_collection(collection_name)
            document = collection.find_one({"_id": ObjectId(id)})
            if document:
                document['_id'] = str(document['_id'])
                return {'statusCode': 200, 'data': document}
            return {'statusCode': 404, 'error': 'Document not found'}
        except Exception as e:
            logger.error(f"Error getting document by id: {e}")
            return {'statusCode': 500, 'error': str(e)}

    '''
    Update document by id
    '''
    def update_by_id(self, id, update_dict, collection_name):
        try:
            collection = self.get_collection(collection_name)
            result = collection.update_one({"_id": ObjectId(id)}, {"$set": update_dict})

            if result.matched_count > 0:
                return {'statusCode': 200, 'message': 'Successfully updated'}

            return {'statusCode': 404, 'error': 'Document not found'}
        except Exception as e:
            logger.error(f"Error updating document by id: {e}")
            return {'statusCode': 500, 'error': str(e)}

    '''
    Update document by filter
    '''
    def update(self, filter_dict, update_dict, collection_name):
        try:
            collection = self.get_collection(collection_name)
            result = collection.update_one(filter_dict, {"$set": update_dict}, upsert=True)
            if result.upserted_id or result.modified_count > 0:
                return {'statusCode': 200, 'message': 'Successfully updated'}
            return {'statusCode': 404, 'error': 'Document not found'}
        except Exception as e:
            logger.error(f"Error updating document by filter: {e}")
            return {'statusCode': 500, 'error': str(e)}

    '''
    Delete document by id
    '''
    def delete_by_id(self, id, collection_name):
        try:
            collection = self.get_collection(collection_name)
            result = collection.delete_one({"_id": ObjectId(id)})
            if result.deleted_count > 0:
                return {'statusCode': 200, 'message': 'Successfully deleted'}
            return {'statusCode': 404, 'error': 'Document not found'}
        except Exception as e:
            logger.error(f"Error deleting document by id: {e}")
            return {'statusCode': 500, 'error': str(e)}

    '''
    Delete document by filter
    '''
    def delete(self, filter_dict, collection_name):
        try:
            collection = self.get_collection(collection_name)
            result = collection.delete_one(filter_dict)
            if result.deleted_count > 0:
                return {'statusCode': 200, 'message': 'Successfully deleted'}
            return {'statusCode': 404, 'error': 'Document not found'}
        except Exception as e:
            logger.error(f"Error deleting document by filter: {e}")
            return {'statusCode': 500, 'error': str(e)}

    '''
    Count documents in collection
    '''
    def count_documents(self, collection_name):
        try:
            collection = self.get_collection(collection_name)
            count = collection.count_documents({})
            return {'statusCode': 200, 'count': count}
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            return {'statusCode': 500, 'error': str(e)}

    '''
    Create user in collection with uid
    '''
    def create_user(self, first_name, last_name, email, password, canViewExplanations):
        try:
            collection = self.get_collection('users')
            last_user = collection.find_one(sort=[('uid', -1)])

            if last_user and 'uid' in last_user:
                last_uid_num = int(last_user['uid'].replace('UID', '').replace('uid', ''))
            else:
                last_uid_num = 0

            new_uid = f"UID{last_uid_num+1:03d}"
            user_doc = {
                'uid': new_uid,
                'firstName': first_name,
                'lastName': last_name,
                'email': email,
                'password': password,
                'canViewExplanations': canViewExplanations
            }
            result = collection.insert_one(user_doc)
            if result.acknowledged:
                user_doc['_id'] = str(result.inserted_id)
                return {'statusCode': 200, 'data': user_doc}
            else:
                return {'statusCode': 400, 'error': 'User could not be added'}
        except Exception as e:
            logger.error(f"Error adding user: {e}")
            return {'statusCode': 500, 'error': str(e)}

    '''
    Check user credentials
    '''
    def check_user_credentials(self, email, password):
        try:
            collection = self.get_collection('users')
            user = collection.find_one({'email': email, 'password': password})
            if user:
                user.pop('password', None)
                user['_id'] = str(user['_id'])
                if 'canViewExplanations' not in user:
                    user['canViewExplanations'] = False
                return {'statusCode': 200, 'data': user}
            else:
                return {'statusCode': 401, 'error': 'Invalid credentials'}
        except Exception as e:
            logger.error(f"Error checking user credentials: {e}")
            return {'statusCode': 500, 'error': str(e)}