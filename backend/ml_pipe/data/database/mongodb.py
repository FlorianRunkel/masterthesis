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
    Initialisierung & Verbindungsaufbau
    '''
    def __init__(self, user='florianrunkel', password='ur04mathesis', db_name='Database', url='cluster0.1lgrrsw.mongodb.net'):
        self.user = user
        self.password = password
        self.db_name = db_name
        self.url = url
        self.client = None
        self.db = None

    def get_mongo_client(self):
        '''Stellt die Verbindung zur MongoDB her und gibt das DB-Objekt zurück.'''
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
                logger.error(f"MongoDB Konfigurationsfehler: {e}")
                raise
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                logger.error(f"Verbindung zu MongoDB fehlgeschlagen: {e}")
                raise
        return self.db

    def get_collection(self, collection_name):
        '''Gibt die Collection zurück.'''
        db = self.get_mongo_client()
        return db[collection_name]

    '''
    CRUD-Methoden
    '''
    def create(self, document, collection_name):
        '''Fügt ein neues Dokument in die Collection ein.'''
        try:
            collection = self.get_collection(collection_name)
            result = collection.insert_one(document)
            if result.acknowledged:
                new_document = collection.find_one({'_id': result.inserted_id})
                if new_document and '_id' in new_document:
                    new_document['_id'] = str(new_document['_id'])
                return {'statusCode': 200, 'data': new_document}
            else:
                return {'statusCode': 400, 'error': 'Dokument konnte nicht eingefügt werden'}
        except Exception as e:
            logger.error(f"Fehler beim Einfügen: {e}")
            return {'statusCode': 500, 'error': str(e)}

    def get(self, filter_dict, collection_name):
        '''Findet alle Dokumente, die dem Filter entsprechen.'''
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

    def get_all(self, collection_name):
        '''Findet alle Dokumente in der Collection.'''
        return self.get({}, collection_name)

    def get_by_id(self, id, collection_name):
        '''Findet ein Dokument anhand der ObjectId.'''
        try:
            collection = self.get_collection(collection_name)
            document = collection.find_one({"_id": ObjectId(id)})
            if document:
                document['_id'] = str(document['_id'])
                return {'statusCode': 200, 'data': document}
            return {'statusCode': 404, 'error': 'Nicht gefunden'}
        except Exception as e:
            logger.error(f"Fehler beim Suchen nach ID: {e}")
            return {'statusCode': 500, 'error': str(e)}

    def update_by_id(self, id, update_dict, collection_name):
        '''Aktualisiert ein Dokument anhand der ObjectId.'''
        try:
            collection = self.get_collection(collection_name)
            result = collection.update_one({"_id": ObjectId(id)}, {"$set": update_dict})
            
            # Korrekte Prüfung: Wurde das Dokument gefunden?
            if result.matched_count > 0:
                return {'statusCode': 200, 'message': 'Erfolgreich aktualisiert'}
            
            # Nur wenn kein Dokument gefunden wurde, geben wir 404 zurück.
            return {'statusCode': 404, 'error': 'Benutzer mit dieser ID nicht gefunden'}
        except Exception as e:
            logger.error(f"Fehler beim Aktualisieren: {e}")
            return {'statusCode': 500, 'error': str(e)}

    def update(self, filter_dict, update_dict, collection_name):
        '''Aktualisiert ein Dokument anhand eines Filters.'''
        try:
            collection = self.get_collection(collection_name)
            result = collection.update_one(filter_dict, {"$set": update_dict}, upsert=True)
            if result.upserted_id or result.modified_count > 0:
                return {'statusCode': 200, 'message': 'Erfolgreich aktualisiert'}
            return {'statusCode': 404, 'error': 'Nicht gefunden'}
        except Exception as e:
            logger.error(f"Fehler beim Aktualisieren: {e}")
            return {'statusCode': 500, 'error': str(e)}

    def delete_by_id(self, id, collection_name):
        '''Löscht ein Dokument anhand der ObjectId.'''
        try:
            collection = self.get_collection(collection_name)
            result = collection.delete_one({"_id": ObjectId(id)})
            if result.deleted_count > 0:
                return {'statusCode': 200, 'message': 'Erfolgreich gelöscht'}
            return {'statusCode': 404, 'error': 'Nicht gefunden'}
        except Exception as e:
            logger.error(f"Fehler beim Löschen: {e}")
            return {'statusCode': 500, 'error': str(e)}

    def delete(self, filter_dict, collection_name):
        '''Löscht ein Dokument anhand eines Filters.'''
        try:
            collection = self.get_collection(collection_name)
            result = collection.delete_one(filter_dict)
            if result.deleted_count > 0:
                return {'statusCode': 200, 'message': 'Erfolgreich gelöscht'}
            return {'statusCode': 404, 'error': 'Nicht gefunden'}
        except Exception as e:
            logger.error(f"Fehler beim Löschen: {e}")
            return {'statusCode': 500, 'error': str(e)}

    '''
    Hilfsmethoden
    '''
    def count_documents(self, collection_name):
        '''Zählt die Anzahl der Dokumente in der Collection.'''
        try:
            collection = self.get_collection(collection_name)
            count = collection.count_documents({})
            return {'statusCode': 200, 'count': count}
        except Exception as e:
            logger.error(f"Fehler beim Zählen: {e}")
            return {'statusCode': 500, 'error': str(e)}

    def create_user(self, first_name, last_name, email, password, canViewExplanations):
        '''Fügt einen neuen User in die Collection users ein. UID wird automatisch vergeben.'''
        try:
            collection = self.get_collection('users')
            # UID generieren: höchster existierender Wert + 1
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
                return {'statusCode': 400, 'error': 'User konnte nicht eingefügt werden'}
        except Exception as e:
            logger.error(f"Fehler beim User-Einfügen: {e}")
            return {'statusCode': 500, 'error': str(e)}

    def check_user_credentials(self, email, password):
        '''Prüft, ob ein User mit E-Mail und Passwort existiert. Gibt Userdaten ohne Passwort zurück.'''
        try:
            collection = self.get_collection('users')
            user = collection.find_one({'email': email, 'password': password})
            if user:
                user.pop('password', None)
                user['_id'] = str(user['_id'])
                # Sicherstellen, dass der Key existiert, um Fehler im Frontend zu vermeiden
                if 'canViewExplanations' not in user:
                    user['canViewExplanations'] = False
                return {'statusCode': 200, 'data': user}
            else:
                return {'statusCode': 401, 'error': 'Falsche Anmeldedaten'}
        except Exception as e:
            logger.error(f"Fehler beim User-Login: {e}")
            return {'statusCode': 500, 'error': str(e)}