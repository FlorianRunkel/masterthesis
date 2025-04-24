import json
import logging
import pandas as pd
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta

import sys
sys.path.insert(0, '/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/')

from backend.ml_pipe.data.database.mongodb import MongoDb
from backend.ml_pipe.data.featureEngineering.featureEngineering import featureEngineering

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_date(date_str):
    """
    Parst verschiedene Datumsformate und gibt ein datetime-Objekt zurück
    """
    if not date_str or date_str == 'Present':
        return None
    
    try:
        # Versuche verschiedene Datumsformate
        formats = ['%d/%m/%Y', '%m/%Y', '%Y']
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
                
        # Wenn kein Format passt, extrahiere das Jahr
        if date_str.isdigit() and len(date_str) == 4:
            return datetime.strptime(date_str, '%Y')
            
        return None
    except Exception:
        return None

def extract_year(date_str):
    """
    Extrahiert das Jahr aus verschiedenen Datumsformaten
    Unterstützt: MM/DD/YYYY, MM/YYYY, YYYY
    """
    if not date_str:
        return None
    try:
        # Versuche verschiedene Datumsformate
        if '/' in date_str:
            parts = date_str.split('/')
            if len(parts) == 3:  # Format: MM/DD/YYYY
                return int(parts[2])
            elif len(parts) == 2:  # Format: MM/YYYY
                return int(parts[1])
        else:  # Format: YYYY oder andere
            # Extrahiere die ersten 4 Ziffern als Jahr
            year_match = ''.join(filter(str.isdigit, date_str))[:4]
            if year_match and len(year_match) == 4:
                return int(year_match)
        return None
    except (ValueError, IndexError):
        return None

def estimate_age_category(profile_info):
    """
    Schätzt die Alterskategorie basierend auf dem frühesten Eintrag in Bildung oder Berufserfahrung
    
    Kategorien:
    1: Early Career (20-25 Jahre)
    2: Early-Mid Career (26-30 Jahre)
    3: Mid Career (31-35 Jahre)
    4: Established Career (36-45 Jahre)
    5: Senior Career (45+ Jahre)
    """
    current_year = datetime.now().year
    earliest_year = None
    
    # Prüfe Bildungseinträge
    for edu in profile_info.get('education', []):
        year = extract_year(edu.get('startDate', ''))
        if year and (earliest_year is None or year < earliest_year):
            earliest_year = year
    
    # Prüfe Berufserfahrung
    for exp in profile_info.get('workExperience', []):
        year = extract_year(exp.get('startDate', ''))
        if year and (earliest_year is None or year < earliest_year):
            earliest_year = year
    
    if not earliest_year:
        return None
    
    # Annahme: Erste Einträge beginnen typischerweise mit ~18 Jahren
    estimated_birth_year = earliest_year - 18
    estimated_age = current_year - estimated_birth_year
    
    # Kategorisierung
    if estimated_age <= 25:
        return 1  # Early Career
    elif estimated_age <= 30:
        return 2  # Early-Mid Career
    elif estimated_age <= 35:
        return 3  # Mid Career
    elif estimated_age <= 45:
        return 4  # Established Career
    else:
        return 5  # Senior Career

def normalize_position(position):
    """Normalisiert die Position für besseres Matching"""
    if not position:
        return ""
    return position.lower().strip()

def extract_career_data(profile_info, fe):
    """Extrahiert und bereinigt die Karrieredaten mit detaillierten Zeitangaben"""
    career_history = []
    
    for exp in profile_info.get('workExperience', []):
        # Verarbeite die Datumsangaben
        start_date = exp.get('startDate', '')
        end_date = exp.get('endDate', '')
        
        # Parse die Datumsangaben
        start_datetime = parse_date(start_date)
        end_datetime = datetime.now() if end_date == 'Present' else parse_date(end_date)
        
        # Berechne zeitbezogene Features
        duration_months = 0
        time_since_start = 0
        time_until_end = 0
        
        if start_datetime and end_datetime:
            duration = relativedelta(end_datetime, start_datetime)
            duration_months = duration.years * 12 + duration.months
            
            # Zeit seit Beginn der Position (in Monaten)
            time_since_start = relativedelta(datetime.now(), start_datetime)
            time_since_start = time_since_start.years * 12 + time_since_start.months
            
            # Zeit bis zum Ende der Position (in Monaten), 0 für aktuelle Position
            if end_date != 'Present':
                time_until_end = relativedelta(datetime.now(), end_datetime)
                time_until_end = time_until_end.years * 12 + time_until_end.months
        
        # Normalisiere und matche die Position
        position = normalize_position(exp.get('position', ''))
        level, branche = fe.find_best_match(position)
        
        career_entry = {
            'position': position,
            'company': exp.get('company', ''),
            'location': exp.get('location', ''),
            'start_date': start_date,
            'end_date': end_date,
            'duration_months': duration_months,
            'time_since_start': time_since_start,
            'time_until_end': time_until_end,
            'is_current': end_date == 'Present',
            'level': level,
            'branche': branche
        }
        career_history.append(career_entry)
    
    # Sortiere nach Startdatum (neueste zuerst)
    career_history.sort(key=lambda x: parse_date(x['start_date']) if parse_date(x['start_date']) else datetime.min, reverse=True)
    
    return career_history

def extract_education_data(profile_info):
    """Extrahiert und bereinigt die Bildungsdaten mit detaillierten Zeitangaben"""
    education_data = []
    
    for edu in profile_info.get('education', []):
        # Verarbeite die Datumsangaben
        start_date = edu.get('startDate', '')
        end_date = edu.get('endDate', '')
        
        edu_entry = {
            'institution': edu.get('institution', ''),
            'degree': edu.get('degree', ''),
            'location': edu.get('location', ''),
            'start_date': start_date,
            'end_date': end_date,
            'duration': edu.get('duration', '')
        }
        education_data.append(edu_entry)
    
    return education_data

def extract_additional_features(career_history, education_data, fe, age_category):
    """
    Extrahiert zusätzliche Features aus dem Karriereverlauf und der Bildung.
    Verarbeitet die tatsächliche Sequenzlänge ohne künstliches Auffüllen.
    """
    features = {}
    
    # Anzahl der Stationen
    features['total_positions'] = len(career_history)
    
    # Zeitbezogene Features für die Sequenz - keine feste Länge mehr
    features['career_sequence'] = []
    for entry in career_history:  # Alle Positionen verwenden
        sequence_entry = {
            'level': entry['level'],
            'branche': entry['branche'],
            'duration_months': entry['duration_months'],
            'time_since_start': entry['time_since_start'],
            'time_until_end': entry['time_until_end'],
            'is_current': 1 if entry['is_current'] else 0
        }
        features['career_sequence'].append(sequence_entry)
    
    # Firmenwechsel und Standorte analysieren
    companies = set()
    locations = set()
    total_duration_months = 0
    position_durations = []
    
    for entry in career_history:
        companies.add(entry['company'])
        if entry['location']:
            locations.add(entry['location'])
        total_duration_months += entry['duration_months']
        position_durations.append(entry['duration_months'])
    
    # Firmenwechsel und Stationen
    features['company_changes'] = len(companies) - 1
    features['total_experience_years'] = round(total_duration_months / 12, 2)
    
    # Standortwechsel
    features['location_changes'] = len(locations) - 1 if len(locations) > 0 else 0
    features['unique_locations'] = len(locations)
    
    # Durchschnittliche Verweildauer pro Position
    features['avg_position_duration_months'] = (
        sum(position_durations) / len(position_durations)
        if position_durations else 0
    )
    
    # Höchster Bildungsabschluss
    degree_ranking = {
        'phd': 5,
        'master': 4,
        'bachelor': 3,
        'apprenticeship': 2,
        'other': 1
    }
    
    highest_degree = 1
    for edu in education_data:
        degree = edu['degree'].lower()
        if 'phd' in degree or 'doktor' in degree:
            highest_degree = max(highest_degree, degree_ranking['phd'])
        elif 'master' in degree or 'msc' in degree or 'mba' in degree:
            highest_degree = max(highest_degree, degree_ranking['master'])
        elif 'bachelor' in degree or 'bsc' in degree or 'ba' in degree:
            highest_degree = max(highest_degree, degree_ranking['bachelor'])
        elif 'apprenticeship' in degree or 'ausbildung' in degree:
            highest_degree = max(highest_degree, degree_ranking['apprenticeship'])
    
    features['highest_degree'] = highest_degree
    
    # Aktuelle Position
    if career_history:
        current_position = career_history[0]  # Erste Position ist die aktuellste
        features['current_position'] = {
            'level': current_position['level'],
            'branche': current_position['branche'],
            'duration_months': current_position['duration_months'],
            'time_since_start': current_position['time_since_start']
        }
    
    # Alterskategorie (mit Fallback auf 0)
    features['age_category'] = age_category if age_category is not None else 0
    
    return features

def import_candidates_from_csv(csv_file):
    try:
        # MongoDB-Verbindung herstellen
        mongo = MongoDb()
        
        # Feature Engineering Instanz erstellen
        fe = featureEngineering()
        
        # CSV-Datei einlesen
        df = pd.read_csv(csv_file)
        
        successful_imports = 0
        failed_imports = 0
        skipped_empty_career = 0
        label_ones_count = 0
        
        for _, row in df.iterrows():
            try:
                if 'linkedinProfileInformation' in row and pd.notna(row['linkedinProfileInformation']):
                    # Bereinige die JSON-Daten
                    json_str = row['linkedinProfileInformation']
                    if json_str.startswith('"') and json_str.endswith('"'):
                        json_str = json_str[1:-1]  # Entferne äußere Anführungszeichen
                    
                    # Ersetze doppelte durch einfache Anführungszeichen
                    json_str = json_str.replace('""', '"')
                    
                    try:
                        profile_info = json.loads(json_str)
                    except json.JSONDecodeError as je:
                        logger.error(f"JSON Parsing Fehler: {str(je)}")
                        logger.error(f"Problematische JSON-Daten: {json_str[:100]}...")
                        failed_imports += 1
                        continue
                    
                    # Extrahiere Karriere- und Bildungsdaten
                    career_history = extract_career_data(profile_info, fe)
                    education_data = extract_education_data(profile_info)
                    
                    # Prüfe auf leere Career History
                    if not career_history:
                        skipped_empty_career += 1
                        continue
                    
                    # Schätze Alterskategorie mit Fallback auf 0
                    age_category = estimate_age_category(profile_info)
                    if age_category is None:
                        age_category = 0  # Default-Wert für nicht bestimmbare Alterskategorie
                        logger.info(f"Keine Alterskategorie bestimmbar - setze auf 0")
                    
                    # Extrahiere zusätzliche Features
                    features = extract_additional_features(career_history, education_data, fe, age_category)
                    
                    # Bestimme Wechselbereitschaft basierend auf den Status
                    communication_status = str(row.get('communicationStatus', '')).lower()
                    candidate_status = str(row.get('candidateStatus', '')).lower()
                    
                    label = 1 if (
                        'interviewbooked' in communication_status.replace(' ', '') or 
                        'accepted' in candidate_status or 
                        'interested' in candidate_status
                    ) else 0
                    
                    if label == 1:
                        label_ones_count += 1
                    
                    # Erstelle bereinigte Kandidatendaten
                    candidate_data = {
                        "features": features,
                        "label": label
                    }
                    
                    # Validiere numerische Werte
                    if features['age_category'] is None:
                        features['age_category'] = 0
                    
                    # Speichere in MongoDB
                    result = mongo.create(candidate_data, 'training_data')
                    
                    if result and result.get('statusCode') == 200:
                        successful_imports += 1
                    else:
                        failed_imports += 1
                        logger.error(f"MongoDB Import fehlgeschlagen für Kandidat")
                # break  # Diese Zeile wird entfernt, da sie die Schleife nach dem ersten Eintrag beendet

            except Exception as e:
                logger.error(f"Fehler beim Import des Kandidaten: {str(e)}")
                failed_imports += 1
                
        logger.info(f"CSV-Import abgeschlossen:")
        logger.info(f"- Erfolgreich importiert: {successful_imports}")
        logger.info(f"- Davon mit Label 1 (wechselbereit): {label_ones_count}")
        logger.info(f"- Fehlgeschlagen: {failed_imports}")
        logger.info(f"- Übersprungen (keine Career History): {skipped_empty_career}")
        return successful_imports, failed_imports, skipped_empty_career, label_ones_count
        
    except Exception as e:
        logger.error(f"Fehler beim CSV-Import: {str(e)}")
        return 0, 0, 0, 0

def handler(filename):
    """
    Verarbeitet CSV-Dateien und importiert die Kandidatendaten in die MongoDB.
    Stellt sicher, dass alle Features gültige numerische Werte haben.
    """
    csv_folder = "backend/ml_pipe/data/datafiles/"
    
    if not os.path.exists(csv_folder):
        print(f"Ordner nicht gefunden: {csv_folder}")
        return
        
    file_path = os.path.join(csv_folder, filename)
    if not os.path.exists(file_path):
        print(f"Datei nicht gefunden: {filename}")
        return
        
    print(f"Verarbeite Datei: {filename}")
    
    try:
        # MongoDB-Verbindung herstellen
        mongo = MongoDb()
        
        # Feature Engineering Instanz erstellen
        fe = featureEngineering()
        
        # CSV-Datei einlesen
        df = pd.read_csv(file_path, 
                        sep=';',           # Semikolon als Trennzeichen
                        quoting=3,         # Deaktiviere Quoting
                        encoding='utf-8',   # UTF-8 Encoding
                        on_bad_lines='skip' # Überspringe problematische Zeilen
                        )
        
        successful_imports = 0
        failed_imports = 0
        skipped_empty_career = 0
        label_ones_count = 0
        
        for _, row in df.iterrows():
            try:
                if 'linkedinProfileInformation' in row and pd.notna(row['linkedinProfileInformation']):
                    # Bereinige die JSON-Daten
                    json_str = row['linkedinProfileInformation']
                    if json_str.startswith('"') and json_str.endswith('"'):
                        json_str = json_str[1:-1]  # Entferne äußere Anführungszeichen
                    
                    # Ersetze doppelte durch einfache Anführungszeichen
                    json_str = json_str.replace('""', '"')
                    
                    try:
                        profile_info = json.loads(json_str)
                    except json.JSONDecodeError as je:
                        logger.error(f"JSON Parsing Fehler: {str(je)}")
                        logger.error(f"Problematische JSON-Daten: {json_str[:100]}...")
                        failed_imports += 1
                        continue
                    
                    # Extrahiere Karriere- und Bildungsdaten
                    career_history = extract_career_data(profile_info, fe)
                    education_data = extract_education_data(profile_info)
                    
                    # Prüfe auf leere Career History
                    if not career_history:
                        skipped_empty_career += 1
                        continue
                    
                    # Schätze Alterskategorie mit Fallback auf 0
                    age_category = estimate_age_category(profile_info)
                    if age_category is None:
                        age_category = 0  # Default-Wert für nicht bestimmbare Alterskategorie
                        logger.info(f"Keine Alterskategorie bestimmbar - setze auf 0")
                    
                    # Extrahiere zusätzliche Features
                    features = extract_additional_features(career_history, education_data, fe, age_category)
                    
                    # Bestimme Wechselbereitschaft basierend auf den Status
                    communication_status = str(row.get('communicationStatus', '')).lower()
                    candidate_status = str(row.get('candidateStatus', '')).lower()
                    
                    label = 1 if (
                        'interviewbooked' in communication_status.replace(' ', '') or 
                        'accepted' in candidate_status
                    ) else 0
                    
                    if label == 1:
                        label_ones_count += 1
                    
                    # Erstelle bereinigte Kandidatendaten
                    candidate_data = {
                        "features": features,
                        "label": label
                    }
                    
                    # Validiere numerische Werte
                    if features['age_category'] is None:
                        features['age_category'] = 0
                    
                    # Speichere in MongoDB
                    result = mongo.create(candidate_data, 'training_data')
                    
                    if result and result.get('statusCode') == 200:
                        successful_imports += 1
                    else:
                        failed_imports += 1
                        logger.error(f"MongoDB Import fehlgeschlagen für Kandidat")
                        
            except Exception as e:
                logger.error(f"Fehler beim Import des Kandidaten: {str(e)}")
                failed_imports += 1
        
        # Zusammenfassung ausgeben
        print(f"\nImport abgeschlossen:")
        print(f"- Erfolgreich importiert: {successful_imports}")
        print(f"- Davon mit Label 1 (wechselbereit): {label_ones_count}")
        print(f"- Fehlgeschlagen: {failed_imports}")
        print(f"- Übersprungen (keine Career History): {skipped_empty_career}")
        
    except Exception as e:
        logger.error(f"Fehler beim CSV-Import: {str(e)}")
        print(f"Import fehlgeschlagen: {str(e)}")
        return

if __name__ == "__main__":
    handler('CID129 (3)_clean.csv')  # Änderung des Dateinamens auf die bereinigte Version