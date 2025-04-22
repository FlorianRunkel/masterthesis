import pymongo
from bson import ObjectId
from pymongo.errors import ConfigurationError
import logging
# Initialize logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class MongoDb:
    def __init__(self, user = 'florianrunkel', password='ur04mathesis', db_name='Database', url='cluster0.1lgrrsw.mongodb.net'):
        self.user = user
        self.password = password
        self.db_name = db_name
        self.url = url
        self.client = None
        self.db = None # Standard Collection für Kandidaten

    '''MongoDB-Verbindung'''
    def get_mongo_client(self):

        if self.client is None:
            try:
                mongo_uri = f"mongodb+srv://{self.user}:{self.password}@{self.url}/{self.db_name}?retryWrites=true&w=majority"
                self.client = pymongo.MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
                self.db = self.client[self.db_name]
            except ConfigurationError as e:
                print(f"MongoDB Konfigurationsfehler: {e}")
                raise
        return self.db

    '''Gibt die Collection zurück'''
    def get_collection(self, collection_name):
        db = self.get_mongo_client()
        return db[collection_name]

    '''Erstellt ein Element in der jeweiligen collection, solange es noch nicht existiert'''
    def create(self, event, collection_name): 
        try:
            collection = self.get_collection(collection_name)
            result = collection.insert_one(event)

            if result.acknowledged:
                print('Dokument erfolgreich eingefuegt')
                new_document = collection.find_one({'_id': result.inserted_id})
                return {
                    'statusCode': 200,
                    'body': 'Dokument erfolgreich eingefuegt',
                    'data': new_document
                }
            else:
                print( "Dokument konnte nicht eingefügt werden")
                return {
                    'statusCode': 400,
                    'body': 'Dokument konnte nicht eingefügt werden'
                }

        except Exception as e:
            print(f"Fehler beim Einfügen des Elements in collection {collection_name}: {e}")
            logger.error(f"Fehler beim Einfügen des Elements in collection {collection_name}: {e}")
            raise e

    '''Holt Elemente aus der jeweiligen collection'''
    def get(self, event, collection_name): 
        try: 
            collection = self.get_collection(collection_name)

            responseSearchCursor = collection.find(event)
            responseDocuments = list(responseSearchCursor)

            if not responseDocuments:
                print(f"Item nicht gefunden {event}")
                return None
            else:
                return responseDocuments

        except Exception as e:
            print(f"Fehler beim Abrufen des Items: {e}")
            logger.error(f"Fehler beim Abrufen des Items: {str(e)}")
            raise e

    '''Updated ein Element in der jeweiligen collection'''
    def update(self, event, collection_name): 
        try: 
            collection = self.get_collection(collection_name)
            
            # Filter and update logic
            filter_query = event.get('filter')
            update_query = event.get('update')

            if not filter_query or not update_query:
                raise ValueError("Filter and update document must be provided")
            
            result = collection.update_one(filter_query, update_query, upsert=True)

            if result.upserted_id:
                print("Document successfully inserted")
                # new_document = collection.find_one({'_id': result.upserted_id})
                return {
                    'statusCode': 200,
                    'body': 'Document successfully updated',
                    # 'data': new_document
                }

            if result.matched_count > 0:
                print("Document successfully updated")
                # updated_document = collection.find_one(filter_query)

                return {
                    'statusCode': 200,
                    'body': 'Document successfully updated',
                    # 'data': updated_document
                }
            else:
                print("Item could not be successfully updated")
                return {
                    'statusCode': 400,
                    'body': 'Item could not be successfully updated'
                }
        
        except Exception as e:
            print(f"Fehler beim Updaten des Elements: {e}")
            logger.error(f"Fehler beim Updaten des Elements: {e}")
            raise e
    
    def find_one(self, event, collection_name):
        try:
            collection = self.get_collection(collection_name)
            response = collection.find_one(event)
            return response
        
        except Exception as e:
            print(f"Fehler beim Finden des Elements: {e}")
            logger.error(f"Fehler beim Finden des Elements: {e}")
            raise e
    
    '''Loescht ein Element in der jeweiligen collection'''
    def delete(self, event, collection_name):
        try:
            collection = self.get_collection(collection_name)

            result=collection.delete_one(event)
            print(result)
            
            if result.deleted_count > 0:
                print("Dokument erfolgreich geloescht")
                return {
                    'statusCode': 200,
                    'body': 'Dokument erfolgreich geloescht'
                }
            else:
                print("Dokument nicht gefunden")
                return {
                    'statusCode': 400,
                    'body': 'Dokument nicht gefunden'
                }

        except Exception as e:
            print(f"Fehler beim Loeschen der Dokumente in {collection_name}: {e}")
            logger.error(f"Fehler beim Loeschen der Dokumente in {collection_name}: {e}")
            raise e

    '''Holt alle Elemente aus der jeweiligen collection'''
    def get_all(self, collection_name):
        try:
            collection = self.get_collection(collection_name)
            cursor = collection.find({})
            documents = list(cursor)
            
            # Konvertiere ObjectId zu String für JSON-Serialisierung
            for doc in documents:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
            
            return {
                'statusCode': 200,
                'data': documents,
                'count': len(documents)
            }
            
        except Exception as e:
            print(f"Fehler beim Laden aller Dokumente aus {collection_name}: {e}")
            logger.error(f"Fehler beim Laden aller Dokumente aus {collection_name}: {e}")
            return {
                'statusCode': 500,
                'error': str(e)
            }