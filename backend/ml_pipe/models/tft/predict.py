import torch
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet, NaNLabelEncoder
from rapidfuzz import process, fuzz

def read_linkedin_profile(profile_path):
    """
    Liest ein LinkedIn-Profil aus einer CSV-Datei ein.
    
    Args:
        profile_path (str): Pfad zur CSV-Datei mit dem LinkedIn-Profil
        
    Returns:
        dict: Dictionary mit den Berufserfahrungen im Format:
        {
            "workExperience": [
                {
                    "position": str,
                    "company": str,
                    "startDate": str,
                    "endDate": str
                },
                ...
            ]
        }
    """
    try:
        # Wenn profile_path ein String ist und wie ein JSON aussieht, parsen wir es direkt
        if isinstance(profile_path, str) and profile_path.strip().startswith('{'):
            profile_json = json.loads(profile_path)
        else:
            # Ansonsten lesen wir die CSV-Datei
            df = pd.read_csv(profile_path)
            profile_json = json.loads(df.iloc[0]['profile_data'])
        
        # Berufserfahrungen extrahieren und formatieren
        work_experiences = []
        for exp in profile_json.get('workExperience', []):
            # Datum-Format anpassen
            start_date = exp['startDate']
            if '/' in start_date:
                # Extrahiere nur Monat und Jahr
                parts = start_date.split('/')
                if len(parts) >= 2:
                    # Wenn das Format DD/MM/YYYY ist, nehmen wir den zweiten Teil als Monat
                    if len(parts) == 3:
                        month, year = parts[1], parts[2]
                    else:
                        month, year = parts[0], parts[1]
                    start_date = f"{month}/{year}"
            
            end_date = exp['endDate']
            if end_date != "Present" and '/' in end_date:
                # Extrahiere nur Monat und Jahr
                parts = end_date.split('/')
                if len(parts) >= 2:
                    # Wenn das Format DD/MM/YYYY ist, nehmen wir den zweiten Teil als Monat
                    if len(parts) == 3:
                        month, year = parts[1], parts[2]
                    else:
                        month, year = parts[0], parts[1]
                    end_date = f"{month}/{year}"
            
            # Neue formatierte Erfahrung hinzufügen
            work_experiences.append({
                "position": exp['position'],
                "company": exp['company'],
                "startDate": start_date,
                "endDate": end_date
            })
        
        return {"workExperience": work_experiences}
    
    except Exception as e:
        print(f"Fehler beim Einlesen des Profils: {str(e)}")
        return None

def parse_date(date_str):
    """
    Hilfsfunktion zum Parsen von Datumsstrings in verschiedenen Formaten.
    
    Args:
        date_str (str): Datumsstring im Format "DD/MM/YYYY" oder "MM/YYYY"
        
    Returns:
        datetime: Parsed datetime object
    """
    if date_str == "Present":
        return datetime.now()
        
    parts = date_str.split('/')
    if len(parts) == 3:  # DD/MM/YYYY
        return datetime(int(parts[2]), int(parts[1]), 1)
    elif len(parts) == 2:  # MM/YYYY
        return datetime(int(parts[1]), int(parts[0]), 1)
    else:
        raise ValueError(f"Ungültiges Datumsformat: {date_str}")

def predict_next_job_change(profile_data):
    """
    Macht eine Vorhersage für den nächsten Jobwechsel basierend auf dem LinkedIn-Profil.
    
    Args:
        profile_data (dict oder str): Dictionary mit den Berufserfahrungen oder JSON-String
        
    Returns:
        list: Liste mit den Vorhersagen für jeden Zeitpunkt
    """
    try:
        # Wenn profile_data ein String ist, parsen wir es zu einem Dictionary
        if isinstance(profile_data, str):
            profile_data = json.loads(profile_data)
        
        # 1. Trainingsdataset laden mit weights_only=True für Sicherheit
        training_dataset = torch.load(
            "/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/models/tft/saved_models/training_dataset.pt",
        )
        
        # 2. Position Mapping laden
        with open("/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/data/featureEngineering/position_level.json", "r") as f:
            position_entries = json.load(f)
        
        # Dict: Position (klein) -> (Level, Branche)
        position_map = {entry["position"].lower(): (entry["level"], entry["branche"]) for entry in position_entries}
        all_positions = list(position_map.keys())
        
        # Level-Zahl auf String
        level_map = {
            1: "Entry", 2: "Junior", 3: "Professional", 4: "Senior", 5: "Lead", 6: "Manager", 7: "Director", 8: "C-Level"
        }
        
        def map_position_fuzzy(pos, threshold=30):
            pos_clean = pos.lower().strip()
            if pos_clean in position_map:
                level, branche = position_map[pos_clean]
                match = pos_clean
                score = 100
            else:
                match, score, _ = process.extractOne(pos_clean, all_positions, scorer=fuzz.ratio)
                if score >= threshold:
                    level, branche = position_map[match]
                else:
                    # Erstelle eine detaillierte Fehlermeldung
                    error_msg = {
                        "error": "Position nicht gefunden",
                        "position": pos,
                        "details": f"Die Position '{pos}' konnte nicht in der Datenbank gefunden werden.",
                        "recommendation": "Bitte wählen Sie eine der folgenden Optionen:",
                        "options": [
                            "1. Wählen Sie eine ähnliche Position aus der Liste:",
                            *[f"   - {p}" for p in all_positions[:5]],  # Zeige die ersten 5 verfügbaren Positionen
                            "2. Geben Sie die Position manuell ein",
                            "3. Wählen Sie eine generische Position (z.B. 'Professional', 'Senior', etc.)"
                        ]
                    }
                    raise ValueError(json.dumps(error_msg))
            level_str = level_map.get(level, str(level)) if isinstance(level, int) else str(level)
            return (match, level_str, branche)
        
        # 3. Daten für Vorhersage vorbereiten
        time_points = []
        experiences = sorted(
            profile_data['workExperience'],
            key=lambda x: parse_date(x['startDate']),
            reverse=True
        )
        
        # Überprüfe, ob genügend Erfahrungen vorhanden sind
        if len(experiences) < 2:
            raise ValueError("Mindestens zwei Berufserfahrungen sind für eine Vorhersage erforderlich.")
        
        # Erstelle zusätzliche Zeitpunkte für längere Zeitreihe
        min_required_points = 24  # Mindestanzahl an Zeitpunkten
        
        for i, exp in enumerate(experiences):
            try:
                start_date = parse_date(exp['startDate'])
                end_date = parse_date(exp['endDate'])
                
                # Position mappen
                try:
                    mapped_pos, level_str, branche = map_position_fuzzy(exp['position'])
                except ValueError as ve:
                    try:
                        error_data = json.loads(str(ve))
                        print(f"\nFehler bei der Position '{exp['position']}':")
                        print(f"Details: {error_data['details']}")
                        print("\nEmpfehlungen:")
                        for option in error_data['options']:
                            print(option)
                    except:
                        print(f"\nFehler bei der Position '{exp['position']}':")
                        print(str(ve))
                    return None
                
                # Erstelle mehr Zeitpunkte pro Position
                points_per_exp = max(8, min_required_points // len(experiences))
                for j in range(points_per_exp):
                    timepoint = start_date + timedelta(days=int((end_date - start_date).days * (j + 1) / points_per_exp))
                    
                    # Berechne Features
                    berufserfahrung = (timepoint - parse_date(experiences[-1]['startDate'])).days
                    anzahl_wechsel = sum(1 for e in experiences if e['endDate'] != "Present" and parse_date(e['endDate']) <= timepoint)
                    anzahl_jobs = sum(1 for e in experiences if parse_date(e['startDate']) <= timepoint)
                    
                    # Berechne durchschnittliche Jobdauer
                    dauer_liste = []
                    for e in experiences:
                        s = parse_date(e['startDate'])
                        e_date = parse_date(e['endDate'])
                        if s < e_date and e_date <= timepoint:
                            dauer_liste.append((e_date - s).days)
                    durchschnittsdauer = sum(dauer_liste) / len(dauer_liste) if dauer_liste else 0
                    
                    # Erstelle DataFrame-Zeile
                    row = {
                        "profile_id": "predict_profile",
                        "time_idx": i * points_per_exp + j,
                        "label": 0,
                        "berufserfahrung_bis_zeitpunkt": berufserfahrung,
                        "anzahl_wechsel_bisher": anzahl_wechsel,
                        "anzahl_jobs_bisher": anzahl_jobs,
                        "durchschnittsdauer_bisheriger_jobs": durchschnittsdauer,
                        "zeitpunkt": timepoint.timestamp(),
                        "aktuelle_position": exp['position'],
                        "mapped_position": mapped_pos,
                        "level_str": level_str,
                        "branche": branche,
                        "weekday": timepoint.weekday(),
                        "weekday_sin": np.sin(2 * np.pi * timepoint.weekday() / 7),
                        "weekday_cos": np.cos(2 * np.pi * timepoint.weekday() / 7),
                        "month": timepoint.month,
                        "month_sin": np.sin(2 * np.pi * timepoint.month / 12),
                        "month_cos": np.cos(2 * np.pi * timepoint.month / 12)
                    }
                    time_points.append(row)
            except ValueError as ve:
                print(f"Fehler beim Verarbeiten der Erfahrung: {str(ve)}")
                continue
        
        if not time_points:
            print("Keine gültigen Zeitpunkte gefunden")
            return None
            
        # 4. DataFrame erstellen
        df = pd.DataFrame(time_points)
        
        # Überprüfe, ob genügend Zeitpunkte vorhanden sind
        if len(df) < min_required_points:
            print(f"Warnung: Zu wenige Zeitpunkte ({len(df)}). Mindestens {min_required_points} erforderlich.")
            # Erstelle zusätzliche Zeitpunkte durch Interpolation
            additional_points_needed = min_required_points - len(df)
            if additional_points_needed > 0:
                # Dupliziere die letzten Zeitpunkte
                last_points = df.iloc[-additional_points_needed:].copy()
                last_points['time_idx'] = range(len(df), len(df) + additional_points_needed)
                df = pd.concat([df, last_points])
        
        # Konvertiere time_idx zu Integer
        df["time_idx"] = df["time_idx"].astype(int)
        
        # Konvertiere andere numerische Spalten zu float
        numeric_columns = [
            "label",
            "berufserfahrung_bis_zeitpunkt",
            "anzahl_wechsel_bisher",
            "anzahl_jobs_bisher",
            "durchschnittsdauer_bisheriger_jobs",
            "zeitpunkt",
            "weekday",
            "weekday_sin",
            "weekday_cos",
            "month",
            "month_sin",
            "month_cos"
        ]
        
        for col in numeric_columns:
            df[col] = df[col].astype(float)
        
        # 5. Vorhersage-Dataset erstellen
        prediction_dataset = TimeSeriesDataSet.from_dataset(
            training_dataset,
            df,
            predict=True,
            stop_randomization=True,
            target_normalizer=None
        )
        
        # 6. Modell laden
        model = TemporalFusionTransformer.load_from_checkpoint("/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/models/tft/saved_models/tft_20250515_163355.ckpt")
        
        # 7. Vorhersage machen
        dataloader = prediction_dataset.to_dataloader(train=False, batch_size=1)
        predictions = model.predict(dataloader)
        
        # 8. Ergebnisse formatieren
        results = []
        for i, pred in enumerate(predictions):
            # Die Werte sind bereits in Tagen
            median_tage = float(pred[0])
            untere_schranke_tage = float(pred[1])
            obere_schranke_tage = float(pred[5])
            
            results.append({
                "tag": i + 1,
                "vorhersage": {
                    "median": median_tage,
                    "untere_schranke": untere_schranke_tage,
                    "obere_schranke": obere_schranke_tage,
                    "unsicherheit": obere_schranke_tage - untere_schranke_tage
                }
            })
            
            print(f"\nVorhersage für Tag {i+1}:")
            print(f"  Median: {median_tage:.1f} Tage ({median_tage/30:.1f} Monate)")
            print(f"  Untere Schranke: {untere_schranke_tage:.1f} Tage ({untere_schranke_tage/30:.1f} Monate)")
            print(f"  Obere Schranke: {obere_schranke_tage:.1f} Tage ({obere_schranke_tage/30:.1f} Monate)")
            print(f"  Unsicherheit: {obere_schranke_tage - untere_schranke_tage:.1f} Tage")
            
            # Interpretation
            if median_tage < 30:
                print(f"  Interpretation: Sehr wahrscheinlicher Jobwechsel innerhalb des nächsten Monats")
            elif median_tage < 90:
                print(f"  Interpretation: Wahrscheinlicher Jobwechsel innerhalb der nächsten 3 Monate")
            elif median_tage < 180:
                print(f"  Interpretation: Möglicher Jobwechsel innerhalb der nächsten 6 Monate")
            else:
                print(f"  Interpretation: Jobwechsel in weiterer Zukunft (> 6 Monate)")
        
        return results
    
    except Exception as e:
        print(f"Fehler bei der Vorhersage: {str(e)}")
        return None

def predict(profile_data, with_llm_explanation=False):
    """
    Hauptfunktion für die Vorhersage, die die gleiche Schnittstelle wie andere Modelle verwendet.
    
    Args:
        profile_data (dict): Dictionary mit den Profildaten
        with_llm_explanation (bool): Wird für Kompatibilität mit anderen Modellen verwendet, hat hier keine Auswirkung
        
    Returns:
        dict: Dictionary mit den Vorhersageergebnissen im Format:
        {
            'confidence': [float],
            'recommendations': [str],
            'status': str,
            'explanations': list,
            'predictions': list  # Zusätzliche detaillierte Vorhersagen
        }
    """
    try:
        # Vorhersage mit der spezifischen TFT-Funktion machen
        predictions = predict_next_job_change(profile_data)
        
        if predictions is None:
            return {
                'error': 'Fehler bei der Vorhersage',
                'status': 'error'
            }
        
        # Median extrahieren (erster Wert der ersten Vorhersage)
        median_days = predictions[0]['vorhersage']['median'] if predictions and 'vorhersage' in predictions[0] else None
        
        # Generiere Empfehlungen basierend auf der ersten Vorhersage
        first_prediction = predictions[0]
        # median_days = first_prediction['vorhersage']['median']
        if median_days is not None:
            if median_days < 30:
                recommendation = "Sehr wahrscheinlicher Jobwechsel innerhalb des nächsten Monats"
            elif median_days < 90:
                recommendation = "Wahrscheinlicher Jobwechsel innerhalb der nächsten 3 Monate"
            elif median_days < 180:
                recommendation = "Möglicher Jobwechsel innerhalb der nächsten 6 Monate"
            else:
                recommendation = "Jobwechsel in weiterer Zukunft (> 6 Monate)"
        else:
            recommendation = "Keine valide Vorhersage möglich."
        
        return {
            'confidence': [median_days],
            'recommendations': [recommendation],
            'status': 'success',
            'explanations': [
                f"Basierend auf Ihrer Berufserfahrung wird ein Jobwechsel in etwa {median_days/30:.1f} Monaten erwartet." if median_days is not None else "Keine valide Vorhersage möglich.",
                f"Die Vorhersage basiert auf dem Medianwert der Modellprognose."
            ],
            'predictions': predictions  # Behalte die detaillierten Vorhersagen bei
        }
        
    except Exception as e:
        print(f"Fehler bei der Vorhersage: {str(e)}")
        return {
            'error': str(e),
            'status': 'error'
        }