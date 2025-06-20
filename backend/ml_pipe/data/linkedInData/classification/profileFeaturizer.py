import json
import logging
import pandas as pd
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
from random import randint

import sys
sys.path.insert(0, '/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/')

from backend.ml_pipe.data.database.mongodb import MongoDb
from backend.ml_pipe.data.featureEngineering.xgboost.feature_engineering_xgb import FeatureEngineering

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_date(date_str):
    if not date_str or date_str == 'Present':
        return None
    try:
        formats = ['%d/%m/%Y', '%m/%Y', '%Y']
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        if date_str.isdigit() and len(date_str) == 4:
            return datetime.strptime(date_str, '%Y')
        return None
    except Exception:
        return None

def normalize_position(position):
    if not position:
        return ""
    return position.lower().strip()

def extract_year(date_str):
    if not date_str:
        return None
    try:
        if '/' in date_str:
            parts = date_str.split('/')
            if len(parts) == 3:
                return int(parts[2])
            elif len(parts) == 2:
                return int(parts[1])
        else:
            year_match = ''.join(filter(str.isdigit, date_str))[:4]
            if year_match and len(year_match) == 4:
                return int(year_match)
        return None
    except (ValueError, IndexError):
        return None

def estimate_age_category(profile_info):
    current_year = datetime.now().year
    earliest_year = None
    for edu in profile_info.get('education', []):
        year = extract_year(edu.get('startDate', ''))
        if year and (earliest_year is None or year < earliest_year):
            earliest_year = year
    for exp in profile_info.get('workExperience', []):
        year = extract_year(exp.get('startDate', ''))
        if year and (earliest_year is None or year < earliest_year):
            earliest_year = year
    if not earliest_year:
        return None
    estimated_birth_year = earliest_year - 18
    estimated_age = current_year - estimated_birth_year
    if estimated_age <= 25:
        return 1
    elif estimated_age <= 30:
        return 2
    elif estimated_age <= 35:
        return 3
    elif estimated_age <= 45:
        return 4
    else:
        return 5

def extract_career_data(profile_info, fe):
    career_history = []
    for exp in profile_info.get('workExperience', []):
        start_date = exp.get('startDate', '')
        end_date = exp.get('EndDate', '') if 'EndDate' in exp else exp.get('endDate', '')
        start_datetime = parse_date(start_date)
        end_datetime = datetime.now() if end_date == 'Present' else parse_date(end_date)
        duration_months = 0
        time_since_start = 0
        time_until_end = 0
        if start_datetime and end_datetime:
            duration = relativedelta(end_datetime, start_datetime)
            duration_months = duration.years * 12 + duration.months
            time_since_start = relativedelta(datetime.now(), start_datetime)
            time_since_start = time_since_start.years * 12 + time_since_start.months
            if end_date != 'Present':
                time_until_end = relativedelta(datetime.now(), end_datetime)
                time_until_end = time_until_end.years * 12 + time_until_end.months
        position = normalize_position(exp.get('position', ''))
        try:
            level, branche, durchschnittszeit_tage = fe.find_best_match(position)
        except Exception:
            print(f"Position {position} kann nicht gemappt werden")
            continue  # Position kann nicht gemappt werden, überspringen
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
            'branche': branche,
            'durchschnittszeit_tage': durchschnittszeit_tage
        }
        career_history.append(career_entry)
    career_history.sort(key=lambda x: parse_date(x['start_date']) if parse_date(x['start_date']) else datetime.min, reverse=True)
    return career_history

def extract_education_data(profile_info):
    education_data = []
    for edu in profile_info.get('education', []):
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
    features = {}
    features['total_positions'] = len(career_history)
    features['career_sequence'] = []
    for entry in career_history:
        sequence_entry = {
            'level': entry['level'],
            'branche': entry['branche'],
            'duration_months': entry['duration_months'],
            'time_since_start': entry['time_since_start'],
            'time_until_end': entry['time_until_end'],
            'is_current': 1 if entry['is_current'] else 0
        }
        features['career_sequence'].append(sequence_entry)
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
    features['company_changes'] = len(companies) - 1
    features['total_experience_years'] = round(total_duration_months / 12, 2)
    features['location_changes'] = len(locations) - 1 if len(locations) > 0 else 0
    features['unique_locations'] = len(locations)
    features['avg_position_duration_months'] = (
        sum(position_durations) / len(position_durations)
        if position_durations else 0
    )
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
    if career_history:
        current_position = career_history[0]
        features['current_position'] = {
            'level': current_position['level'],
            'branche': current_position['branche'],
            'duration_months': current_position['duration_months'],
            'time_since_start': current_position['time_since_start']
        }
    features['age_category'] = age_category
    return features

def generate_career_sequences(career_history, education_data, fe, age_category):
    # Aggregierte Features für die gesamte Karriere
    highest_degree = extract_additional_features(career_history, education_data, fe, age_category)['highest_degree']

    sequence_examples = []
    for i, pos in enumerate(career_history):
        if pos['is_current']:
            continue
        current_sequence = career_history[i:]
        companies = set(p['company'] for p in current_sequence)
        locations = set(p['location'] for p in current_sequence if p['location'])
        total_duration_months = sum(p['duration_months'] for p in current_sequence)

        total_experience_days = int(total_duration_months * 30.44)
        company_changes = len(companies) - 1
        location_changes = len(locations) - 1 if len(locations) > 0 else 0
        current_pos = current_sequence[0]
        duration = int(current_pos['duration_months']*30.44) 
        
        # Vorherige Positionen als Features (nur die Positionen vor der aktuellen)
        career_history_features = {}
        for j, prev_pos in enumerate(reversed(career_history[i+1:])):
            career_history_features[f"position_{j+1}"] = {
                "duration": int(prev_pos['duration_months'] * 30.44),
                "branche": prev_pos['branche'],
                "level": prev_pos['level'],
                "position": prev_pos['position']
            }
        
        # Basis-Features
        career_history_durations = sum(
            v["duration"] for v in career_history_features.values()
        )
        total_experience_days = career_history_durations + duration
        sequence_features = {
            "company_changes": company_changes,
            "total_experience_days": total_experience_days,
            "location_changes": location_changes,
            #"average_months_per_position": avg_months_per_position,
            "highest_degree": highest_degree,
            "position": current_pos['position'],
            "position_level": current_pos['level'],
            "position_branche": current_pos['branche'],
            "position_duration": int(duration),
            "avg_position_duration_days": current_pos.get('durchschnittszeit_tage', None),
            "age_category": age_category,   
            "career_history": career_history_features
        }
            
        label = 1
        sequence_examples.append({
            "features": sequence_features,
            "label": label
        })
        
        if duration > 1:

            career_history_durations = sum(
                v["duration"] for v in career_history_features.values()
            )
            
            random_tag = randint(1, duration - 1)
            random_pos = dict(current_pos)
            random_pos['duration'] = random_tag
            total_experience_days = career_history_durations + int(random_pos['duration'])

            sequence_features_random = {
                "company_changes": company_changes,
                "total_experience_days": total_experience_days,
                "location_changes": location_changes,
                #"average_months_per_position": avg_months_per_position,
                "highest_degree": highest_degree,
                "position": current_pos['position'],
                "position_level": current_pos['level'],
                "position_branche": current_pos['branche'],
                "position_duration": int(random_pos['duration']),
                "avg_position_duration_days": current_pos.get('durchschnittszeit_tage', None),
                "age_category": age_category,
                "career_history": career_history_features
            }
                
            sequence_examples.append({
                "features": sequence_features_random,
                "label": 0
            })

    return sequence_examples

def is_valid_sequence(seq_features):
    if seq_features.get("position_level", 0) == 0 or seq_features.get("position_branche", 0) == 0:
        return False
    if seq_features.get("age_category", 0) == 0:
        return False
    return True

def import_candidates_from_csv(csv_file):
    mongo = MongoDb()
    fe = FeatureEngineering()
    logger.info("Lese CSV-Datei...")
    df = pd.read_csv(csv_file)
    logger.info(f"CSV geladen. {len(df)} Zeilen gefunden.")
    successful_imports = 0
    failed_imports = 0
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
                age_category = estimate_age_category(profile_info)
                if not career_history or len(career_history) < 1 or age_category is None:
                    continue
                sequence_examples = generate_career_sequences(career_history, education_data, fe, age_category)
                for seq in sequence_examples:
                    if is_valid_sequence(seq["features"]):
                        result = mongo.create(seq, 'classification_dataset')
                        if result and result.get('statusCode') == 200:
                            successful_imports += 1
                        else:
                            failed_imports += 1
            else:
                logger.info(f"Profil {index} übersprungen (keine Profildaten)")

        except Exception as e:
            logger.error(f"Fehler beim Import des Kandidaten {index}: {str(e)}")
            failed_imports += 1
    logger.info(f"Import abgeschlossen: Erfolgreich: {successful_imports}, Fehlgeschlagen: {failed_imports}")
    return successful_imports, failed_imports

def handler(file_path):
    '''
    Verarbeitet CSV-Dateien und importiert die Kandidatendaten in die MongoDB.
    '''
    logger.info(f"Starte Verarbeitung der Datei: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"Datei nicht gefunden: {file_path}")
        return
    logger.info(f"Verarbeite Datei: {file_path}")
    try:
        # Rufe die Import-Funktion auf
        successful, failed = import_candidates_from_csv(file_path)
        logger.info("\nFinaler Import-Status:")
        logger.info(f"✓ Erfolgreich importierte Datenpunkte: {successful}")
        logger.info(f"✗ Fehlgeschlagene Imports: {failed}")
    except Exception as e:
        logger.error(f"Kritischer Fehler beim Verarbeiten der Datei: {str(e)}")
        return

'''
csv_folder = "backend/ml_pipe/data/datafiles/"
if not os.path.exists(csv_folder):
    logger.error(f"Ordner nicht gefunden: {csv_folder}")
else:
    for filename in os.listdir(csv_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(csv_folder, filename)
            handler(file_path)
'''