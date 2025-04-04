import sqlite3
import pandas as pd

class DataCleaningHandler:
    def __init__(self, db_path='ml_pipe/data/database/linkedin_data.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)

    def clean_profiles(self):
        df = pd.read_sql_query("SELECT * FROM profiles", self.conn)

        # Duplikate anhand Vorname, Nachname, Position, Firma
        df_clean = df.drop_duplicates(subset=['first_name', 'last_name', 'current_position', 'current_company'])

        # Leere Felder mit Standardwerten oder NaN füllen
        df_clean = df_clean.fillna({
            'location': 'Unknown',
            'industry': 'Unknown',
            'experience_years': 0,
            'connections_count': 0
        })

        # Alte Tabelle löschen und neue speichern
        df_clean.to_sql('profiles', self.conn, if_exists='replace', index=False)
        print(f"✓ Tabelle 'profiles' bereinigt – {len(df)} → {len(df_clean)} Einträge")

    def clean_experiences(self):
        df = pd.read_sql_query("SELECT * FROM experiences", self.conn)
        df_clean = df.drop_duplicates(subset=['profile_id', 'company', 'position', 'start_date'])
        df_clean = df_clean.dropna(subset=['company', 'position', 'start_date'])

        df_clean.to_sql('experiences', self.conn, if_exists='replace', index=False)
        print(f"✓ Tabelle 'experiences' bereinigt – {len(df)} → {len(df_clean)} Einträge")

    def clean_education(self):
        df = pd.read_sql_query("SELECT * FROM education", self.conn)
        df_clean = df.drop_duplicates(subset=['profile_id', 'institution', 'degree'])
        df_clean = df_clean.dropna(subset=['institution', 'degree'])

        df_clean.to_sql('education', self.conn, if_exists='replace', index=False)
        print(f"✓ Tabelle 'education' bereinigt – {len(df)} → {len(df_clean)} Einträge")

    """
    Führt die Bereinigung für alle Tabellen durch
    """
    def clean_all(self):
        self.clean_profiles()
        self.clean_experiences()
        self.clean_education()

    def close(self):
        self.conn.close()