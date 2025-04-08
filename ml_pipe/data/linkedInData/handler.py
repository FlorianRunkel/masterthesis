import json
import logging
import pandas as pd
import os
from datetime import datetime
import random

import sys
sys.path.insert(0, '/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/')

from ml_pipe.data.database.mongodb import MongoDb

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
Importiert Kandidatendaten aus einer CSV-Datei in MongoDB.
Speichert nur Education- und CareerHistory-Daten.
"""
def import_candidates_from_csv(csv_file):
    try:
        # MongoDB-Verbindung herstellen
        mongo = MongoDb()

        # CSV-Datei einlesen
        logger.info(f"Lese CSV-Datei: {csv_file}")
        df = pd.read_csv(csv_file)

        # Zähler für erfolgreiche und fehlgeschlagene Imports
        successful_imports = 0
        failed_imports = 0
        skipped_empty_career = 0

        # Jeden Kandidaten verarbeiten
        for _, row in df.iterrows():
            try:
                # linkedinProfileInformation extrahieren und parsen
                if 'linkedinProfileInformation' in row and pd.notna(row['linkedinProfileInformation']):
                    try:
                        profile_info = json.loads(row['linkedinProfileInformation'])

                        # Kandidaten-ID generieren (falls nicht vorhanden)
                        candidate_id = str(random.randint(100000000, 999999999))

                        # CareerHistory-Daten extrahieren
                        career_history = profile_info.get('workExperience', [])

                        # Prüfen, ob die Career History leer ist
                        if not career_history:
                            logger.warning(f"Kandidat übersprungen: {candidate_id} - Keine Career History gefunden")
                            skipped_empty_career += 1
                            continue

                        # Prüfen, ob der Kandidat bereits existiert
                        existing_candidate = mongo.find_one({"candidate_id": candidate_id}, 'CareerData')

                        # Education-Daten extrahieren
                        education_data = profile_info.get('education', [])

                        # Kandidatendaten erstellen
                        candidate_data = {
                            "candidate_id": candidate_id,
                            "education": education_data,
                            "career_history": career_history
                        }

                        if existing_candidate:
                            # Update der existierenden Kandidatendaten
                            # Behalte nur Education und CareerHistory bei
                            update_data = {
                                "education": education_data,
                                "career_history": career_history,
                                "last_updated": datetime.now().isoformat()
                            }

                            # Update durchführen
                            update_result = mongo.update(
                                {
                                    'filter': {"candidate_id": candidate_id},
                                    'update': {"$set": update_data}
                                },
                                'CareerData'
                            )

                            if update_result and update_result.get('statusCode') == 200:
                                successful_imports += 1
                                logger.info(f"Kandidat erfolgreich aktualisiert: {candidate_id}")
                            else:
                                failed_imports += 1
                                logger.warning(f"Kandidat konnte nicht aktualisiert werden: {candidate_id}")
                        else:
                            # Create durchführen
                            result = mongo.create(candidate_data, 'CareerData')

                            if result and result.get('statusCode') == 200:
                                successful_imports += 1
                                logger.info(f"Kandidat erfolgreich importiert: {candidate_id}")
                            else:
                                failed_imports += 1
                                logger.warning(f"Kandidat konnte nicht importiert werden: {candidate_id}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Fehler beim Parsen der linkedinProfileInformation: {str(e)}")
                        failed_imports += 1
                else:
                    logger.warning("Keine linkedinProfileInformation gefunden")
                    failed_imports += 1

            except Exception as e:
                logger.error(f"Fehler beim Import des Kandidaten: {str(e)}")
                failed_imports += 1

        logger.info(f"CSV-Import abgeschlossen: {successful_imports} erfolgreich, {failed_imports} fehlgeschlagen, {skipped_empty_career} übersprungen (leere Career History)")
        return successful_imports, failed_imports, skipped_empty_career

    except Exception as e:
        logger.error(f"Fehler beim CSV-Import: {str(e)}")
        return 0, 0, 0

import os
csv_folder = "ml_pipe/data/datafiles/"

if os.path.exists(csv_folder):
    all_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

    total_successful = 0
    total_failed = 0
    total_skipped = 0

    for filename in all_files:
        file_path = os.path.join(csv_folder, filename)
        print(f"Verarbeite Datei: {filename}")
        successful, failed, skipped = import_candidates_from_csv(file_path)
        total_successful += successful
        total_failed += failed
        total_skipped += skipped

    print(f"\nImport abgeschlossen: {total_successful} erfolgreich, {total_failed} fehlgeschlagen, {total_skipped} übersprungen (leere Career History)")

else:
    print(f"Ordner nicht gefunden: {csv_folder}")