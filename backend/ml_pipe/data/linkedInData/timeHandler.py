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
        print(profile_info)
        experiences = profile_info.get('workExperience', [])
        profile_id = profile_info.get('linkedinProfile', None)
        if profile_id and isinstance(profile_id, str):
            # Nur den letzten Teil der URL extrahieren
            profile_id = profile_id.rstrip('/').split('/')[-1]
        else:
            # Fallback: Name + Zufallszahl
            name = profile_info.get('firstName', 'unknown') + profile_info.get('lastName', '')
            profile_id = f"{name.lower().replace(' ', '-')}-{random.randint(1000,9999)}"
    except Exception as e:
        print("Fehler beim Parsen:", e)
        return []

    experiences = sorted(
        experiences,
        key=lambda x: parse_date(x.get('startDate', '')) or datetime(1900, 1, 1),
        reverse=True
    )

    samples = []
    for idx in range(len(experiences) - 1):
        current_exp = experiences[idx]
        current_start = parse_date(current_exp.get('startDate', ''))
        current_end = parse_date(current_exp.get('endDate', ''))

        if not current_start or not current_end or current_start >= current_end:
            continue

        positionsdauer = (current_end - current_start).days
        prozentpunkte = [0.25, 0.5, 0.75, 0.9]

        for p in prozentpunkte:
            sample_timepoint = current_start + timedelta(days=int(positionsdauer * p))
            if sample_timepoint < current_start or sample_timepoint > current_end:
                continue
            label_days = (current_end - sample_timepoint).days

            # a) Berufserfahrung bis zum Zeitpunkt
            alle_starts = [parse_date(exp.get('startDate', '')) for exp in experiences if parse_date(exp.get('startDate', '')) and parse_date(exp.get('startDate', '')) < sample_timepoint]
            if alle_starts:
                berufserfahrung_bis_zeitpunkt = (sample_timepoint - min(alle_starts)).days
            else:
                berufserfahrung_bis_zeitpunkt = 0

            # b) Anzahl Wechsel bisher (Anzahl Positionen mit Enddatum < sample_timepoint)
            anzahl_wechsel_bisher = sum(1 for exp in experiences if parse_date(exp.get('endDate', '')) and parse_date(exp.get('endDate', '')) < sample_timepoint)

            # c) Anzahl Jobs bisher (Anzahl Positionen mit Startdatum < sample_timepoint)
            anzahl_jobs_bisher = sum(1 for exp in experiences if parse_date(exp.get('startDate', '')) and parse_date(exp.get('startDate', '')) < sample_timepoint)

            # d) Durchschnittsdauer bisheriger Jobs
            dauer_liste = []
            for exp in experiences:
                s = parse_date(exp.get('startDate', ''))
                e = parse_date(exp.get('endDate', ''))
                if s and e and e < sample_timepoint and s < e:
                    dauer_liste.append((e - s).days)
            durchschnittsdauer_bisheriger_jobs = sum(dauer_liste) / len(dauer_liste) if dauer_liste else 0

            samples.append({
                "profile_id": profile_id,
                "aktuelle_position": current_exp.get("position", ""),
                "zeitpunkt": sample_timepoint.timestamp(),
                "label": label_days,
                "berufserfahrung_bis_zeitpunkt": berufserfahrung_bis_zeitpunkt,
                "anzahl_wechsel_bisher": anzahl_wechsel_bisher,
                "anzahl_jobs_bisher": anzahl_jobs_bisher,
                "durchschnittsdauer_bisheriger_jobs": durchschnittsdauer_bisheriger_jobs,
            })
 
    return samples

if __name__ == "__main__":
    df = pd.read_csv("/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/data/datafiles/CID261.csv")

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
        mongo_db.create(sample, 'time_dataset')
    print(f"{len(all_samples)} Einträge erfolgreich in 'time_dataset' gespeichert.")
