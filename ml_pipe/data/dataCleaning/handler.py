import logging
import sys
import re
from datetime import datetime
sys.path.insert(0, '/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/')

from ml_pipe.data.database.mongodb import MongoDb

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
Bereinigt die Karrierdaten in der MongoDB.
Prüft auf unvollständige oder fehlende Daten und aktualisiert sie.
"""
def clean_data(collection_name='CareerData'):
    try:
        mongo = MongoDb()
        collection_name = collection_name
        # Alle Kandidaten aus der MongoDB abrufen
        logger.info("Rufe alle Kandidaten aus der MongoDB ab...")
        candidates = mongo.get({}, collection_name)

        if not candidates:
            logger.warning("Keine Kandidaten in der Datenbank gefunden.")
            return 0, 0

        logger.info(f"{len(candidates)} Kandidaten gefunden.")

        # Duplikate identifizieren und entfernen
        unique_candidates = remove_duplicates(candidates)
        logger.info(f"{len(candidates) - len(unique_candidates)} Duplikate entfernt.")

        # Zähler für erfolgreiche und fehlgeschlagene Updates
        successful_updates = 0
        failed_updates = 0

        # Jeden Kandidaten verarbeiten
        for candidate in unique_candidates:
            try:
                candidate_id = candidate.get('candidate_id')
                if not candidate_id:
                    logger.warning("Kandidat ohne ID gefunden, überspringe...")
                    failed_updates += 1
                    continue

                # Prüfen, ob die Career History vorhanden und nicht leer ist
                career_history = candidate.get('career_history', [])
                if not career_history:
                    logger.warning(f"Kandidat {candidate_id} hat keine Career History, überspringe...")
                    failed_updates += 1
                    continue

                # Prüfen, ob die Education-Daten vorhanden sind
                education_data = candidate.get('education', [])

                # Bereinige Career History
                cleaned_career_history = clean_career_history(career_history)

                # Bereinige Education-Daten
                cleaned_education = clean_education_data(education_data)

                # Bereinigte Daten erstellen
                cleaned_data = {
                    "candidate_id": candidate_id,
                    "education": cleaned_education,
                    "career_history": cleaned_career_history
                }

                # Update durchführen
                update_result = mongo.update(
                    {
                        'filter': {"candidate_id": candidate_id},
                        'update': {"$set": cleaned_data}
                    },
                    collection_name
                )

                if update_result and update_result.get('statusCode') == 200:
                    successful_updates += 1
                    logger.info(f"Kandidat erfolgreich aktualisiert: {candidate_id}")
                else:
                    failed_updates += 1
                    logger.warning(f"Kandidat konnte nicht aktualisiert werden: {candidate_id}")

            except Exception as e:
                logger.error(f"Fehler beim Bereinigen des Kandidaten: {str(e)}")
                failed_updates += 1

        logger.info(f"Bereinigung abgeschlossen: {successful_updates} erfolgreich aktualisiert, {failed_updates} fehlgeschlagen")
        return successful_updates, failed_updates

    except Exception as e:
        logger.error(f"Fehler bei der Bereinigung der Karrierdaten: {str(e)}")
        return 0, 0

"""
Entfernt Duplikate aus der Kandidatenliste.
Duplikate werden anhand der Career History identifiziert.
"""
def remove_duplicates(candidates):

    unique_candidates = []
    seen_career_histories = set()

    for candidate in candidates:
        career_history = candidate.get('career_history', [])
        if not career_history:
            continue

        # Erstelle einen Hash der Career History für den Vergleich
        career_hash = hash_career_history(career_history)

        if career_hash not in seen_career_histories:
            seen_career_histories.add(career_hash)
            unique_candidates.append(candidate)

    return unique_candidates

"""
Erstellt einen Hash der Career History für den Vergleich.
"""
def hash_career_history(career_history):
    # Extrahiere die wichtigsten Informationen aus der Career History
    career_info = []
    for job in career_history:
        company = job.get('company', '').lower()
        position = job.get('position', '').lower()
        start_date = job.get('startDate', '')
        end_date = job.get('endDate', '')

        career_info.append(f"{company}|{position}|{start_date}|{end_date}")

    # Sortiere die Informationen, um die Reihenfolge zu ignorieren
    career_info.sort()

    # Erstelle einen String aus den sortierten Informationen
    return "||".join(career_info)

"""
Bereinigt die Career History-Daten.
- Standardisiert Datumsformate
- Konvertiert alle Texte in Kleinbuchstaben
- Ersetzt fehlende Werte mit Standardwerten
"""
def clean_career_history(career_history):

    cleaned_history = []

    for job in career_history:
        # Standardisiere Datumsformate
        start_date = standardize_date(job.get('startDate', ''))
        end_date = standardize_date(job.get('endDate', ''))

        # Konvertiere Texte in Kleinbuchstaben
        company = job.get('company', '').lower()
        position = job.get('position', '').lower()
        location = job.get('location', '').lower()
        description = job.get('description', '').lower()

        # Ersetze fehlende Werte mit Standardwerten
        company = replace_missing_value(company, "unbekanntes unternehmen")
        position = replace_missing_value(position, "unbekannte position")
        location = replace_missing_value(location, "unbekannter ort")

        # Erstelle bereinigtes Job-Objekt
        cleaned_job = {
            "company": company,
            "position": position,
            "location": location,
            "startDate": start_date,
            "endDate": end_date,
            "description": description
        }

        # Füge zusätzliche Felder hinzu, falls vorhanden
        for key, value in job.items():
            if key not in cleaned_job and value is not None:
                if isinstance(value, str):
                    cleaned_job[key] = value.lower()
                else:
                    cleaned_job[key] = value

        cleaned_history.append(cleaned_job)

    return cleaned_history

"""
Bereinigt die Education-Daten.
- Standardisiert Datumsformate
- Konvertiert alle Texte in Kleinbuchstaben
- Ersetzt fehlende Werte mit Standardwerten
"""
def clean_education_data(education_data):

    cleaned_education = []

    for edu in education_data:
        # Standardisiere Datumsformate
        start_date = standardize_date(edu.get('startDate', ''))
        end_date = standardize_date(edu.get('endDate', ''))

        # Konvertiere Texte in Kleinbuchstaben
        institution = edu.get('institution', '').lower()
        degree = edu.get('degree', '').lower()
        field_of_study = edu.get('subjectStudy', '').lower()
        location = edu.get('location', '').lower()

        # Ersetze fehlende Werte mit Standardwerten
        institution = replace_missing_value(institution, "unbekannte institution")
        degree = replace_missing_value(degree, "unbekannter abschluss")
        field_of_study = replace_missing_value(field_of_study, "unbekanntes fach")
        location = replace_missing_value(location, "unbekannter ort")

        # Erstelle bereinigtes Education-Objekt
        cleaned_edu = {
            "institution": institution,
            "degree": degree,
            "field_of_study": field_of_study,
            "location": location,
            "startDate": start_date,
            "endDate": end_date
        }

        # Füge zusätzliche Felder hinzu, falls vorhanden
        for key, value in edu.items():
            if key not in cleaned_edu and value is not None:
                if isinstance(value, str):
                    cleaned_edu[key] = value.lower()
                else:
                    cleaned_edu[key] = value

        cleaned_education.append(cleaned_edu)

    return cleaned_education

"""
Standardisiert Datumsformate.
Konvertiert verschiedene Datumsformate in das Format YYYY-MM-DD.
"""
def standardize_date(date_str):

    if not date_str or date_str.lower() in ['present', 'jetzt', 'aktuell', 'heute']:
        return datetime.now().strftime('%Y-%m-%d')

    # Entferne alle nicht-numerischen Zeichen außer Punkten und Bindestrichen
    date_str = re.sub(r'[^\d.-]', '', date_str)

    # Versuche verschiedene Datumsformate zu erkennen
    try:
        # Format: DD.MM.YYYY
        if re.match(r'^\d{1,2}\.\d{1,2}\.\d{4}$', date_str):
            parts = date_str.split('.')
            if len(parts) == 3:
                day, month, year = parts
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

        # Format: MM/DD/YYYY
        if re.match(r'^\d{1,2}/\d{1,2}/\d{4}$', date_str):
            parts = date_str.split('/')
            if len(parts) == 3:
                month, day, year = parts
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

        # Format: YYYY-MM-DD
        if re.match(r'^\d{4}-\d{1,2}-\d{1,2}$', date_str):
            return date_str

        # Wenn kein Format erkannt wird, gib das aktuelle Datum zurück
        return datetime.now().strftime('%Y-%m-%d')
    except:
        # Bei Fehlern gib das aktuelle Datum zurück
        return datetime.now().strftime('%Y-%m-%d')

"""
Ersetzt fehlende Werte mit einem Standardwert.
"""
def replace_missing_value(value, default):
    if not value or value.lower() in ['nan', 'none', 'null', '']:
        return default
    return value

def handler():
    print("Starte Bereinigung aller Collections...")
    successful, failed = clean_data()
    return successful, failed

successful, failed = handler()
print(f"Bereinigung abgeschlossen: {successful} erfolgreich, {failed} fehlgeschlagen")