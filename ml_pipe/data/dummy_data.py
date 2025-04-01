import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import os

def create_dummy_database():
    """Erstellt eine SQLite-Datenbank mit Dummy-LinkedIn-Daten"""
    
    # Erstelle Verzeichnisse falls nicht vorhanden
    os.makedirs('ml_pipe/data/database', exist_ok=True)
    
    # Verbindung zur SQLite-Datenbank herstellen
    conn = sqlite3.connect('ml_pipe/data/database/linkedin_data.db')
    cursor = conn.cursor()
    
    # Tabellen erstellen
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS profiles (
        id INTEGER PRIMARY KEY,
        first_name TEXT,
        last_name TEXT,
        location TEXT,
        industry TEXT,
        current_position TEXT,
        current_company TEXT,
        experience_years INTEGER,
        education_level TEXT,
        connections_count INTEGER,
        created_at TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS experiences (
        id INTEGER PRIMARY KEY,
        profile_id INTEGER,
        company TEXT,
        position TEXT,
        start_date DATE,
        end_date DATE,
        FOREIGN KEY (profile_id) REFERENCES profiles (id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS education (
        id INTEGER PRIMARY KEY,
        profile_id INTEGER,
        institution TEXT,
        degree TEXT,
        start_date DATE,
        end_date DATE,
        FOREIGN KEY (profile_id) REFERENCES profiles (id)
    )
    ''')
    
    # Generiere Dummy-Daten
    n_profiles = 1000
    
    # Basis-Daten für Profile
    profiles_data = {
        'first_name': np.random.choice(['Max', 'Anna', 'Thomas', 'Lisa', 'Michael', 'Sarah', 'David', 'Julia'], n_profiles),
        'last_name': np.random.choice(['Müller', 'Schmidt', 'Weber', 'Meyer', 'Wagner', 'Becker', 'Schulz', 'Hoffmann'], n_profiles),
        'location': np.random.choice(['Berlin', 'München', 'Hamburg', 'Frankfurt', 'Köln', 'Düsseldorf', 'Stuttgart', 'Leipzig'], n_profiles),
        'industry': np.random.choice(['Technology', 'Finance', 'Healthcare', 'Education', 'Manufacturing', 'Retail', 'Consulting'], n_profiles),
        'current_position': np.random.choice(['Entry Level', 'Mid Level', 'Senior Level', 'Lead', 'Manager', 'Director', 'VP', 'C-Level'], n_profiles),
        'current_company': np.random.choice(['TechCorp', 'FinanceBank', 'HealthCare Plus', 'EduTech', 'Manufacturing Co', 'Retail Group', 'Consulting Firm'], n_profiles),
        'experience_years': np.random.randint(0, 30, n_profiles),
        'education_level': np.random.choice(['Bachelor', 'Master', 'PhD', 'MBA'], n_profiles),
        'connections_count': np.random.randint(100, 30000, n_profiles),
        'created_at': [datetime.now() - timedelta(days=np.random.randint(0, 3650)) for _ in range(n_profiles)]
    }
    
    # Füge Profile in die Datenbank ein
    for i in range(n_profiles):
        cursor.execute('''
        INSERT INTO profiles (first_name, last_name, location, industry, current_position, 
                            current_company, experience_years, education_level, connections_count, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            profiles_data['first_name'][i],
            profiles_data['last_name'][i],
            profiles_data['location'][i],
            profiles_data['industry'][i],
            profiles_data['current_position'][i],
            profiles_data['current_company'][i],
            profiles_data['experience_years'][i],
            profiles_data['education_level'][i],
            profiles_data['connections_count'][i],
            profiles_data['created_at'][i]
        ))
        
        profile_id = cursor.lastrowid
        
        # Generiere 1-3 Erfahrungen pro Profil
        n_experiences = np.random.randint(1, 4)
        for _ in range(n_experiences):
            start_date = datetime.now() - timedelta(days=np.random.randint(365, 3650))
            end_date = start_date + timedelta(days=np.random.randint(365, 3650))
            
            cursor.execute('''
            INSERT INTO experiences (profile_id, company, position, start_date, end_date)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                profile_id,
                np.random.choice(['TechCorp', 'FinanceBank', 'HealthCare Plus', 'EduTech', 'Manufacturing Co']),
                np.random.choice(['Junior Developer', 'Senior Developer', 'Project Manager', 'Team Lead', 'Department Head']),
                start_date,
                end_date
            ))
        
        # Generiere 1-2 Bildungsabschlüsse pro Profil
        n_education = np.random.randint(1, 3)
        for _ in range(n_education):
            start_date = datetime.now() - timedelta(days=np.random.randint(3650, 7300))
            end_date = start_date + timedelta(days=np.random.randint(365, 1825))
            
            cursor.execute('''
            INSERT INTO education (profile_id, institution, degree, start_date, end_date)
            VALUES (?, ?, ?, ?, ?)
            ''', (
                profile_id,
                np.random.choice(['University of Technology', 'Business School', 'Technical University', 'University of Applied Sciences']),
                np.random.choice(['Bachelor', 'Master', 'PhD', 'MBA']),
                start_date,
                end_date
            ))
    
    # Änderungen speichern und Verbindung schließen
    conn.commit()
    conn.close()
    
    print("Dummy-Datenbank wurde erstellt: ml_pipe/data/database/linkedin_data.db")

if __name__ == "__main__":
    create_dummy_database() 