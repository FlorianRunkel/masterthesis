import sqlite3

class DatabaseHandler:
    def __init__(self, db_path='ml_pipe/data/database/career_data.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

    """
    Fügt eine neue Tabelle mit dem angegebenen Namen hinzu
    """
    def add_db_table(self, tablename):
        sql = f'''
        CREATE TABLE IF NOT EXISTS {tablename} (
            id INTEGER PRIMARY KEY,
            profile_id INTEGER,
            company TEXT,
            position TEXT,
            start_date DATE,
            end_date DATE
        )
        '''
        self.cursor.execute(sql)
        self.conn.commit()

    """
    Fügt einen neuen Datensatz in die angegebene Tabelle ein
    """
    def add_entry(self, table, data: dict):
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        sql = f'INSERT INTO {table} ({columns}) VALUES ({placeholders})'
        self.cursor.execute(sql, tuple(data.values()))
        self.conn.commit()
        return self.cursor.lastrowid

    """
    Aktualisiert einen Datensatz anhand der ID
    """
    def update_entry(self, table, entry_id, data: dict):
        set_clause = ', '.join([f'{key} = ?' for key in data.keys()])
        sql = f'UPDATE {table} SET {set_clause} WHERE id = ?'
        self.cursor.execute(sql, tuple(data.values()) + (entry_id,))
        self.conn.commit()

    """
    Löscht einen Eintrag anhand der ID
    """
    def delete_entry(self, table, entry_id):
        sql = f'DELETE FROM {table} WHERE id = ?'
        self.cursor.execute(sql, (entry_id,))
        self.conn.commit()
    
    """
    Get Datatset 
    """
    def get_entry(self, table, entry_id):
        sql = f'SELECT * FROM {table} WHERE id = ?'
        self.cursor.execute(sql, (entry_id,))
        return self.cursor.fetchone()

    """
    Schließt Verbindung
    """
    def close(self):
        self.conn.close()
    
'''
db = DatabaseHandler()

# Ein Karriereverlauf für ein anonymes Profil
experience_entry = {
    'profile_id': 1,  # Referenz auf anonymes Profil (z. B. einfach eine ID)
    'company': 'Google',
    'position': 'Software Engineer',
    'start_date': '2019-06-01',
    'end_date': '2022-08-01'
}
db.add_entry('career_history', experience_entry)

# Weitere Einträge für dieselbe ID
db.add_entry('career_history', {
    'profile_id': 1,
    'company': 'Amazon',
    'position': 'Senior Engineer',
    'start_date': '2022-09-01',
    'end_date': None
})

db.close()
'''