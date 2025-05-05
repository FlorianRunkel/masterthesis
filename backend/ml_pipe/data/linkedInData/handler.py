import json
import logging
import pandas as pd
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
import csv
import random

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
    Extrahiert Features aus dem Karriereverlauf und der Bildung.
    Berücksichtigt nur die Karrierehistorie bis zum betrachteten Zeitpunkt.
    """
    features = {}
    
    # Anzahl der Stationen bis zu diesem Zeitpunkt
    features['total_positions'] = len(career_history)
    
    # Zeitbezogene Features für die Sequenz
    features['career_sequence'] = []
    for entry in career_history:  # Nur die Positionen bis zum betrachteten Zeitpunkt
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
    
    # Firmenwechsel und Stationen bis zu diesem Zeitpunkt
    features['company_changes'] = len(companies) - 1
    features['total_experience_years'] = round(total_duration_months / 12, 2)
    
    # Standortwechsel bis zu diesem Zeitpunkt
    features['location_changes'] = len(locations) - 1 if len(locations) > 0 else 0
    features['unique_locations'] = len(locations)
    
    # Durchschnittliche Verweildauer pro Position bis zu diesem Zeitpunkt
    features['avg_position_duration_months'] = (
        sum(position_durations) / len(position_durations)
        if position_durations else 0
    )
    
    # Höchster Bildungsabschluss (bleibt konstant)
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
    
    # Aktuelle Position zum betrachteten Zeitpunkt
    if career_history:
        current_position = career_history[0]  # Erste Position in der Teilsequenz
        features['current_position'] = {
            'level': current_position['level'],
            'branche': current_position['branche'],
            'duration_months': current_position['duration_months'],
            'time_since_start': current_position['time_since_start']
        }
    
    # Alterskategorie (wird für jeden Zeitpunkt neu berechnet)
    features['age_category'] = age_category
    
    return features

def generate_career_sequences(career_history, education_data, fe, age_category):
    """
    Generiert sowohl das vollständige Profil als auch einzelne Teilsequenzen als Trainingsbeispiele.
    Die career_sequence enthält IMMER alle Felder der Karrierestationen.
    Returns:
        tuple: (full_profile, sequence_examples)
    """
    def enrich_entry(entry):
        # Stelle sicher, dass alle relevanten Felder enthalten sind und is_current immer 0 oder 1 ist
        return {
            'level': entry.get('level', 0),
            'branche': entry.get('branche', 0),
            'duration_months': entry.get('duration_months', 0),
            'time_since_start': entry.get('time_since_start', 0),
            'time_until_end': entry.get('time_until_end', 0),
            'is_current': 1 if entry.get('is_current', 0) else 0
        }

    # 1. Erstelle das vollständige Profil
    full_profile = {
        "features": {
            "total_positions": len(career_history),
            "career_sequence": [enrich_entry(e) for e in career_history],
            "company_changes": len(set(pos['company'] for pos in career_history)) - 1,
            "total_experience_years": round(sum(pos['duration_months'] for pos in career_history) / 12, 2),
            "location_changes": len(set(pos['location'] for pos in career_history if pos['location'])) - 1,
            "unique_locations": len(set(pos['location'] for pos in career_history if pos['location'])),
            "avg_position_duration_months": sum(pos['duration_months'] for pos in career_history) / len(career_history),
            "highest_degree": extract_additional_features(career_history, education_data, fe, age_category)['highest_degree'],
            "current_position": {
                "level": career_history[0]['level'],
                "branche": career_history[0]['branche'],
                "duration_months": career_history[0]['duration_months'],
                "time_since_start": career_history[0]['time_since_start']
            },
            "age_category": age_category
        },
        "label": 1
    }
    
    # 2. Erstelle die einzelnen Teilsequenzen
    sequence_examples = []
    for i in range(len(career_history) - 1):
        current_sequence = career_history[:(i+1)]
        sequence_features = {
            "total_positions": len(current_sequence),
            "career_sequence": [enrich_entry(e) for e in current_sequence],
            "company_changes": len(set(pos['company'] for pos in current_sequence)) - 1,
            "total_experience_years": round(sum(pos['duration_months'] for pos in current_sequence) / 12, 2),
            "location_changes": len(set(pos['location'] for pos in current_sequence if pos['location'])) - 1,
            "unique_locations": len(set(pos['location'] for pos in current_sequence if pos['location'])),
            "avg_position_duration_months": sum(pos['duration_months'] for pos in current_sequence) / len(current_sequence),
            "highest_degree": extract_additional_features(current_sequence, education_data, fe, age_category)['highest_degree'],
            "current_position": {
                "level": current_sequence[-1]['level'],
                "branche": current_sequence[-1]['branche'],
                "duration_months": current_sequence[-1]['duration_months'],
                "time_since_start": current_sequence[-1]['time_since_start']
            },
            "age_category": age_category
        }
        sequence_example = {
            "features": sequence_features,
            "label": 1
        }
        sequence_examples.append(sequence_example)
    
    return full_profile, sequence_examples

def is_valid_sequence(seq_features):
    # Prüfe, ob alle Karrierestationen gültige Werte haben
    for entry in seq_features["career_sequence"]:
        if (
            entry["level"] == 0 or
            entry["branche"] == 0 
        ):
            return False
    # Prüfe globale Features
    if seq_features.get("age_category", 0) == 0:
        return False
    return True

def generate_random_negative_sequence(career_history, education_data, fe, age_category, mongo):
    """Erstellt eine zufällige Teilsequenz aus dem vollständigen Profil und gibt sie als declined Sequenz (label=0) zurück."""
    if len(career_history) <= 1:
        return None
    # Wähle eine zufällige Länge für die Teilsequenz (mindestens 1, maximal len(career_history)-1)
    seq_length = random.randint(1, len(career_history) - 1)
    current_sequence = career_history[:seq_length]
    # Passe die Zeitdaten an: Wähle einen zufälligen Zeitpunkt innerhalb der tatsächlichen Dauer
    for entry in current_sequence:
        actual_duration = entry.get('duration_months', 0)
        if actual_duration > 0:
            # Wähle einen zufälligen Zeitpunkt (in Monaten) innerhalb der tatsächlichen Dauer
            random_point = random.randint(0, actual_duration)
            entry['duration_months'] = random_point
            # time_since_start ist die Zeit seit Beginn der Position bis zum zufälligen Zeitpunkt
            entry['time_since_start'] = random_point
            entry['time_until_end'] = actual_duration - random_point
    sequence_features = {
        "total_positions": len(current_sequence),
        "career_sequence": [{
            'level': entry.get('level', 0),
            'branche': entry.get('branche', 0),
            'duration_months': entry.get('duration_months', 0),
            'time_since_start': entry.get('time_since_start', 0),
            'time_until_end': entry.get('time_until_end', 0),
            'is_current': 1 if entry.get('is_current', 0) else 0
        } for entry in current_sequence],
        "company_changes": len(set(pos['company'] for pos in current_sequence)) - 1,
        "total_experience_years": round(sum(pos['duration_months'] for pos in current_sequence) / 12, 2),
        "location_changes": len(set(pos['location'] for pos in current_sequence if pos['location'])) - 1,
        "unique_locations": len(set(pos['location'] for pos in current_sequence if pos['location'])),
        "avg_position_duration_months": sum(pos['duration_months'] for pos in current_sequence) / len(current_sequence),
        "highest_degree": extract_additional_features(current_sequence, education_data, fe, age_category)['highest_degree'],
        "current_position": {
            "level": current_sequence[-1]['level'],
            "branche": current_sequence[-1]['branche'],
            "duration_months": current_sequence[-1]['duration_months'],
            "time_since_start": current_sequence[-1]['time_since_start']
        },
        "age_category": age_category
    }
    if is_valid_sequence(sequence_features):
        return {"features": sequence_features, "label": 0}
    return None

def import_candidates_from_csv(csv_file):
    try:
        mongo = MongoDb()
        fe = featureEngineering()

        logger.info("Lese CSV-Datei...")
        df = pd.read_csv(csv_file)
        logger.info(f"CSV geladen. {len(df)} Zeilen gefunden.")

        # Prüfe die aktuelle Label-Verteilung in der MongoDB
        all_data = mongo.get_all('training_data2').get('data', [])
        label_ones = sum(1 for d in all_data if d.get('label', 0) == 1)
        label_zeros = sum(1 for d in all_data if d.get('label', 0) == 0)
        logger.info(f"Aktuelle Label-Verteilung in MongoDB:")
        logger.info(f"- label=1: {label_ones}")
        logger.info(f"- label=0: {label_zeros}")
        ratio = (label_ones / label_zeros) if label_zeros > 0 else float('inf')
        logger.info(f"- Verhältnis label=1/label=0: {ratio:.2f}")

        # Definiere die erlaubte Ratio (60:40 bis 40:60)
        min_ratio = 0.6
        max_ratio = 1.67
        allow_sequences = (min_ratio <= ratio <= max_ratio) or (label_ones == 0 and label_zeros == 0)
        if allow_sequences:
            logger.info("Label-Verteilung ist ausgeglichen oder leer. Importiere Profile UND Sequenzen.")
        else:
            logger.warning("Label-Verteilung ist unausgeglichen! Importiere NUR Profile, KEINE Sequenzen.")

        successful_imports = 0
        failed_imports = 0
        skipped_empty_career = 0
        total_sequences_processed = 0
        unique_profiles = 0
        profile_ids = []

        for index, row in df.iterrows():
            try:
                if 'linkedinProfileInformation' in row and pd.notna(row['linkedinProfileInformation']):
                    json_str = row['linkedinProfileInformation']
                    try:
                        json_str = json_str.encode('latin1').decode('utf-8')
                    except UnicodeError:
                        pass
                    json_str = ''.join(char for char in json_str if ord(char) >= 32 or char in '\n\r\t')
                    try:
                        profile_info = json.loads(json_str)
                    except json.JSONDecodeError:
                        import re
                        json_str = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', json_str)
                        profile_info = json.loads(json_str)
                    career_history = extract_career_data(profile_info, fe)
                    education_data = extract_education_data(profile_info)
                    if not career_history or len(career_history) < 2:
                        skipped_empty_career += 1
                        continue
                    age_category = estimate_age_category(profile_info)
                    if age_category is None:
                            skipped_empty_career += 1
                            continue
                    candidate_status = str(row.get('candidateStatus', '')).lower()
                    communicationStatus = str(row.get('communicationStatus', '')).lower()
                    is_positive = ('accepted' in candidate_status) or ('interested' in candidate_status) or ('interviewBooked' in communicationStatus)
                    is_negative = ('declined' in candidate_status) or ('futureProspect' in candidate_status) or ('followUpSent' in communicationStatus) or ('secondFollowUpSent' in communicationStatus)
                    # Wenn der Status nan ist, aber gültige Features vorhanden sind, importiere als negativ (label=0)
                    if pd.isna(candidate_status) and is_valid_sequence(full_profile["features"]):
                        is_negative = True
                    full_profile, sequence_examples = generate_career_sequences(
                        career_history, 
                        education_data, 
                        fe, 
                        age_category
                    )
                    # Importiere Profile und ggf. Sequenzen
                    if is_positive and is_valid_sequence(full_profile["features"]):
                        try:
                            profile_to_save = {k: v for k, v in full_profile.items() if k in ["features", "label"]}
                            profile_to_save["label"] = 1
                            result = mongo.create(profile_to_save, 'training_data2')
                            if result and result.get('statusCode') == 200 and result.get('_id'):
                                successful_imports += 1
                                profile_ids.append(result.get('_id'))
                                logger.debug(f"Vollständiges Profil {index} erfolgreich gespeichert")
                            else:
                                failed_imports += 1
                                logger.error(f"Fehler beim Speichern des vollständigen Profils {index}")
                        except Exception as e:
                            logger.error(f"Fehler beim Import des vollständigen Profils {index}: {str(e)}")
                            failed_imports += 1
                        # Sequenzen nur speichern, wenn erlaubt
                        if allow_sequences:
                            for seq_num, sequence_data in enumerate(sequence_examples, 1):
                                if is_valid_sequence(sequence_data["features"]):
                                    try:
                                        # Speichere Sequenz mit label=1
                                        sequence_to_save = {k: v for k, v in sequence_data.items() if k in ["features", "label"]}
                                        sequence_to_save["label"] = 1
                                        result = mongo.create(sequence_to_save, 'training_data2')
                                        if result and result.get('statusCode') == 200:
                                            total_sequences_processed += 1
                                            successful_imports += 1
                                            logger.debug(f"Sequenz {seq_num} von Profil {index} erfolgreich gespeichert")
                                        else:
                                            failed_imports += 1
                                            logger.error(f"Fehler beim Speichern der Sequenz {seq_num} von Profil {index}")
                                    except Exception as e:
                                        logger.error(f"Fehler beim Import der Sequenz {seq_num} von Profil {index}: {str(e)}")
                                        failed_imports += 1
                                    # Erstelle eine declined Sequenz (label=0) für jede Sequenz mit label=1
                                    declined_sequence = generate_random_negative_sequence(career_history, education_data, fe, age_category, mongo)
                                    if declined_sequence:
                                        result = mongo.create(declined_sequence, 'training_data2')
                                        if result and result.get('statusCode') == 200:
                                            total_sequences_processed += 1
                                            successful_imports += 1
                                            logger.debug(f"Declined Sequenz {seq_num} von Profil {index} erfolgreich gespeichert")
                        else:
                                            failed_imports += 1
                                            logger.error(f"Fehler beim Speichern der declined Sequenz {seq_num} von Profil {index}")

                        if sequence_examples:
                            unique_profiles += 1
                            logger.debug(f"Profil {index} mit {len(sequence_examples)} Sequenzen verarbeitet")
                    elif is_negative and is_valid_sequence(full_profile["features"]):
                        try:
                            profile_to_save = {k: v for k, v in full_profile.items() if k in ["features", "label"]}
                            profile_to_save["label"] = 0
                            result = mongo.create(profile_to_save, 'training_data2')
                            if result and result.get('statusCode') == 200:
                                successful_imports += 1
                                profile_ids.append(result.get('_id'))
                                logger.debug(f"Negatives Profil {index} erfolgreich gespeichert")
                            else:
                                failed_imports += 1
                                logger.error(f"Fehler beim Speichern des negativen Profils {index}")
                        except Exception as e:
                            logger.error(f"Fehler beim Import des negativen Profils {index}: {str(e)}")
                        failed_imports += 1
                else:
                        logger.info(f"Profil {index} übersprungen (Status: {candidate_status})")
            except Exception as e:
                logger.error(f"Fehler beim Import des Kandidaten {index}: {str(e)}")
                failed_imports += 1

        # Verifiziere die Imports
        logger.info("\nVerifiziere Imports...")
        try:
            # Prüfe ob alle Profile in der DB sind
            for profile_id in profile_ids:
                if profile_id is None:
                    continue
                result = mongo.find_one({'_id': profile_id}, 'training_data2')
                if not result or result.get('statusCode') != 200:
                    logger.warning(f"Profil {profile_id} konnte nicht in der DB gefunden werden!")
        except Exception as e:
            logger.error(f"Fehler bei der Verifikation: {str(e)}")

        # Detaillierte Zusammenfassung
        logger.info(f"\nCSV-Import abgeschlossen:")
        logger.info(f"- Verarbeitete Profile: {unique_profiles}")
        logger.info(f"- Vollständige Profile importiert: {unique_profiles}")
        logger.info(f"- Zusätzliche Sequenzen importiert: {total_sequences_processed}")
        logger.info(f"- Gesamt Datenpunkte: {unique_profiles + total_sequences_processed}")
        logger.info(f"- Fehlgeschlagen: {failed_imports}")
        logger.info(f"- Übersprungen: {skipped_empty_career}")
        
        # Zähle die Label-Verteilung in der Datenbank
        try:
            all_data = mongo.get_all('training_data2').get('data', [])
            label_ones = sum(1 for d in all_data if d.get('label', 0) == 1)
            label_zeros = sum(1 for d in all_data if d.get('label', 0) == 0)
            logger.info(f"Label-Verteilung in training_data2:")
            logger.info(f"- label=1: {label_ones}")
            logger.info(f"- label=0: {label_zeros}")
        except Exception as e:
            logger.error(f"Fehler beim Auslesen der Label-Verteilung: {str(e)}")
        
        return successful_imports, failed_imports, skipped_empty_career, total_sequences_processed

    except Exception as e:
        logger.error(f"Kritischer Fehler beim CSV-Import: {str(e)}")
        return 0, 0, 0, 0

def handler(filename):
    """
    Verarbeitet CSV-Dateien und importiert die Kandidatendaten in die MongoDB.
    """
    logger.info(f"Starte Verarbeitung der Datei: {filename}")
    
    csv_folder = "backend/ml_pipe/data/datafiles/"
    
    if not os.path.exists(csv_folder):
        logger.error(f"Ordner nicht gefunden: {csv_folder}")
        return
        
        file_path = os.path.join(csv_folder, filename)
    if not os.path.exists(file_path):
        logger.error(f"Datei nicht gefunden: {file_path}")
        return
        
    logger.info(f"Verarbeite Datei: {file_path}")
    
    try:
        # Rufe die Import-Funktion auf
        successful, failed, skipped, sequences = import_candidates_from_csv(file_path)
        
        logger.info("\nFinaler Import-Status:")
        logger.info(f"✓ Erfolgreich importierte Datenpunkte: {successful}")
        logger.info(f"✗ Fehlgeschlagene Imports: {failed}")
        logger.info(f"- Übersprungene Profile: {skipped}")
        logger.info(f"+ Zusätzliche Sequenzen: {sequences}")
        
    except Exception as e:
        logger.error(f"Kritischer Fehler beim Verarbeiten der Datei: {str(e)}")
        return

if __name__ == "__main__":
    logger.info("Starte Import-Prozess...")
    handler('CID103 (1).csv')  # Änderung des Dateinamens auf die bereinigte Version
    logger.info("Import-Prozess beendet.")