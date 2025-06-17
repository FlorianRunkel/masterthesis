import sys
import getpass
from ml_pipe.data.database.mongodb import MongoDb

if __name__ == '__main__':
    print('--- Neuen User anlegen ---')
    first_name = input('Vorname: ').strip()
    last_name = input('Nachname: ').strip()
    email = input('E-Mail: ').strip()
    password = getpass.getpass('Passwort: ').strip()

    if not all([first_name, last_name, email, password]):
        print('Alle Felder sind erforderlich!')
        sys.exit(1)

    db = MongoDb()
    result = db.create_user(first_name, last_name, email, password)
    if result['statusCode'] == 200:
        print('User erfolgreich angelegt:')
        print(result['data'])
    else:
        print('Fehler beim Anlegen:', result.get('error', 'Unbekannter Fehler'))
        sys.exit(1) 