import pandas as pd
import json
from datetime import datetime, timedelta
import random
import sys
import os
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

def extract_year(date_str):
    if not date_str or pd.isna(date_str):
        return None
    if date_str == "Present":
        return None
    for fmt in ["%d/%m/%Y", "%m/%Y", "%Y"]:
        try:
            return datetime.strptime(date_str, fmt).year
        except Exception:
            continue
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

def get_company_info_field(exp, field):
    # Erst direkt im Experience-Objekt prüfen
    if field in exp and exp[field]:
        return exp[field]
    # Dann in companyInformation prüfen
    if 'companyInformation' in exp and isinstance(exp['companyInformation'], dict):
        val = exp['companyInformation'].get(field)
        # Falls industry eine Liste ist, als String joinen
        if field == 'industry' and isinstance(val, list):
            return ', '.join(val)
        return val
    return ""

def categorize_company_size(employee_count):
    if employee_count is None:
        return None
    try:
        count = int(employee_count)
        if count < 10:
            return 'micro'
        elif count < 50:
            return 'small'
        elif count < 250:
            return 'medium'
        elif count < 1000:
            return 'large'
        else:
            return 'enterprise'
    except Exception:
        return None

def process_profile(row):
    try:
        profile_info = json.loads(row['linkedinProfileInformation'])
        print(profile_info)
        experiences = profile_info.get('workExperience', [])
        education_data = profile_info.get('education', [])
        profile_id = profile_info.get('linkedinProfile', None)
        if profile_id and isinstance(profile_id, str):
            profile_id = profile_id.rstrip('/').split('/')[-1]
        else:
            name = profile_info.get('firstName', 'unknown') + profile_info.get('lastName', '')
            profile_id = f"{name.lower().replace(' ', '-')}-{random.randint(1000,9999)}"
        age_category = estimate_age_category(profile_info)
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

            # Degree-Ranking analog zur Klassifikation
            degree_ranking = {
                'phd': 5,
                'master': 4,
                'bachelor': 3,
                'apprenticeship': 2,
                'other': 1
            }
            highest_degree = 1
            for edu in education_data:
                degree = edu.get('degree', '').lower()
                if 'phd' in degree or 'doktor' in degree:
                    highest_degree = max(highest_degree, degree_ranking['phd'])
                elif 'master' in degree or 'msc' in degree or 'mba' in degree:
                    highest_degree = max(highest_degree, degree_ranking['master'])
                elif 'bachelor' in degree or 'bsc' in degree or 'ba' in degree:
                    highest_degree = max(highest_degree, degree_ranking['bachelor'])
                elif 'apprenticeship' in degree or 'ausbildung' in degree:
                    highest_degree = max(highest_degree, degree_ranking['apprenticeship'])

            # anzahl_standortwechsel: Anzahl verschiedener Städte/Länder
            locations = set()
            for exp in experiences:
                loc = exp.get('location', '').strip().lower()
                if loc:
                    locations.add(loc)
            anzahl_standortwechsel = len(locations)

            # study_field: Erster nicht-leerer subjectStudy oder fieldOfStudy aus education_data
            study_field = None
            for edu in education_data:
                val = edu.get('subjectStudy') or edu.get('fieldOfStudy') or edu.get('degree')
                if val and isinstance(val, str) and val.strip():
                    study_field = val.strip()
                    break

            samples.append({
                "profile_id": profile_id,
                "aktuelle_position": current_exp.get("position", ""),
                "zeitpunkt": sample_timepoint.timestamp(),
                "label": label_days,
                "berufserfahrung_bis_zeitpunkt": berufserfahrung_bis_zeitpunkt,
                "anzahl_wechsel_bisher": anzahl_wechsel_bisher,
                "anzahl_jobs_bisher": anzahl_jobs_bisher,
                "durchschnittsdauer_bisheriger_jobs": durchschnittsdauer_bisheriger_jobs,
                "highest_degree": highest_degree,
                "age_category": age_category,
                "anzahl_standortwechsel": anzahl_standortwechsel,
                "study_field": study_field,
                "company_name": get_company_info_field(current_exp, "company"),
                "company_industry": get_company_info_field(current_exp, "industry"),
                "company_location": get_company_info_field(current_exp, "location"),
                "company_size_category": categorize_company_size(get_company_info_field(current_exp, "employee_count")),
            })
 
    return samples

if __name__ == "__main__":
    datafiles_dir = "/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/data/datafiles"
    all_samples = []

    for filename in os.listdir(datafiles_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(datafiles_dir, filename)
            print(f"Verarbeite Datei: {filepath}")
            df = pd.read_csv(filepath)
            for _, row in df.iterrows():
                samples = process_profile(row)
                all_samples.extend(samples)

    print(f"Fertig! {len(all_samples)} Samples werden in MongoDB gespeichert...")

    mongo_db = MongoDb(user='florianrunkel', password='ur04mathesis', db_name='Database')
    for sample in all_samples:
        print(sample)
        mongo_db.create(sample, 'timeseries_dataset')
    print(f"{len(all_samples)} Einträge erfolgreich in 'timeseries_dataset' gespeichert.")
