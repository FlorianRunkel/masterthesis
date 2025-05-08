import pandas as pd
import json
from datetime import datetime, timedelta
import random
import sys
sys.path.insert(0, '/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/')
from backend.ml_pipe.data.database.mongodb import MongoDb

def classify_change(diff_months):
    if diff_months <= 6:
        return 1
    elif diff_months <= 12:
        return 2
    elif diff_months <= 24:
        return 3
    elif diff_months > 24:
        return 4

def parse_date(date_str):
    if not date_str or pd.isna(date_str):
        return None
    if date_str == "Present":
        return datetime.now()
    try:
        return datetime.strptime(date_str, "%d/%m/%Y")
    except Exception:
        try:
            return datetime.strptime(date_str, "%m/%Y")
        except Exception:
            try:
                return datetime.strptime(date_str, "%Y")
            except Exception:
                return None

def random_point_for_category(current_end, cat_min, cat_max, current_start):
    """Wähle einen zufälligen Zeitpunkt, der so weit vor current_end liegt, dass er in die Kategorie fällt und im Positionszeitraum liegt."""
    # cat_min, cat_max in Monaten (z.B. 0-6, 7-12, ...)
    latest = current_end - timedelta(days=cat_min*30)
    earliest = current_end - timedelta(days=cat_max*30)
    if earliest < current_start:
        earliest = current_start
    if latest <= earliest:
        return None
    delta = (latest - earliest).days
    if delta <= 0:
        return None
    random_days = random.randint(0, delta)
    return earliest + timedelta(days=random_days)

def process_profile(row):
    try:
        profile_info = json.loads(row['linkedinProfileInformation'])
        experiences = profile_info.get('workExperience', [])
    except Exception as e:
        print("Fehler beim Parsen:", e)
        return []

    experiences = sorted(
        experiences,
        key=lambda x: parse_date(x.get('startDate', '')) or datetime(1900, 1, 1),
        reverse=True
    )

    # Kategorien: (min, max) Monate
    categories = [
        (0, 6, 1),
        (7, 12, 2),
        (13, 24, 3),
        (25, 600, 4)  # 600 als "unendlich"
    ]

    samples = []
    for idx in range(len(experiences) - 1):
        current_exp = experiences[idx]
        current_start = parse_date(current_exp.get('startDate', ''))
        current_end = parse_date(current_exp.get('endDate', ''))

        if not current_start or not current_end or current_start >= current_end:
            continue

        for cat_min, cat_max, cat_label in categories:
            random_point = random_point_for_category(current_end, cat_min, cat_max, current_start)
            if not random_point or random_point < current_start or random_point > current_end:
                continue
            diff_months = (current_end.year - random_point.year) * 12 + (current_end.month - random_point.month)
            samples.append({
                "aktuelle_position": current_exp.get("position", ""),
                "zeitpunkt": random_point.strftime("%d/%m/%Y"),
                "label": cat_label,
                "wechselzeitraum": diff_months,
            })

    return samples

if __name__ == "__main__":
    df = pd.read_csv("/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/data/datafiles/CID224.csv")

    # Nur das erste Profil testen
    first_row = df.iloc[0]
    samples = process_profile(first_row)

    print(f"Anzahl erzeugter Samples: {len(samples)}")

    all_samples = []
    for _, row in df.iterrows():
        print(row)
        samples = process_profile(row)
        all_samples.extend(samples)

    print(f"Fertig! {len(all_samples)} Samples werden in MongoDB gespeichert...")

    # Mit deiner eigenen MongoDb-Klasse speichern
    mongo_db = MongoDb(user='florianrunkel', password='ur04mathesis', db_name='Database')
    for sample in all_samples:
        print(sample)
        mongo_db.create(sample, 'career_labels_tft')
    print(f"{len(all_samples)} Einträge erfolgreich in 'career_labels_tft' gespeichert.")
