import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import json
from rapidfuzz import process, fuzz

class FeatureEngineering:
    def __init__(self, position_map, all_positions, level_map):
        self.position_map = position_map
        self.all_positions = all_positions
        self.level_map = level_map

    def parse_date(self, date_str):
        if date_str == "Present":
            return datetime.now()
        parts = date_str.split('/')
        if len(parts) == 3:
            return datetime(int(parts[2]), int(parts[1]), 1)
        elif len(parts) == 2:
            return datetime(int(parts[1]), int(parts[0]), 1)
        else:
            raise ValueError(f"Ungültiges Datumsformat: {date_str}")

    def map_position_fuzzy(self, pos, threshold=30):
        pos_clean = pos.lower().strip()
        if pos_clean in self.position_map:
            level, branche = self.position_map[pos_clean]
            match = pos_clean
            score = 100
        else:
            match, score, _ = process.extractOne(pos_clean, self.all_positions, scorer=fuzz.ratio)
            if score >= threshold:
                level, branche = self.position_map[match]
            else:
                error_msg = {
                    "error": "Position nicht gefunden",
                    "position": pos,
                    "details": f"Die Position '{pos}' konnte nicht in der Datenbank gefunden werden.",
                    "recommendation": "Bitte wählen Sie eine der folgenden Optionen:",
                    "options": [
                        "1. Wählen Sie eine ähnliche Position aus der Liste:",
                        *[f"   - {p}" for p in self.all_positions[:5]],
                        "2. Geben Sie die Position manuell ein",
                        "3. Wählen Sie eine generische Position (z.B. 'Professional', 'Senior', etc.)"
                    ]
                }
                raise ValueError(json.dumps(error_msg))
        level_str = self.level_map.get(level, str(level)) if isinstance(level, int) else str(level)
        return (match, level_str, branche)

    def prepare_time_points(self, profile_data, min_required_points=24):
        time_points = []
        experiences = sorted(
            profile_data['workExperience'],
            key=lambda x: self.parse_date(x['startDate']),
            reverse=True
        )
        if len(experiences) < 2:
            raise ValueError("Mindestens zwei Berufserfahrungen sind für eine Vorhersage erforderlich.")
        for i, exp in enumerate(experiences):
            try:
                start_date = self.parse_date(exp['startDate'])
                end_date = self.parse_date(exp['endDate'])
                mapped_pos, level_str, branche = self.map_position_fuzzy(exp['position'])
                points_per_exp = max(8, min_required_points // len(experiences))
                for j in range(points_per_exp):
                    timepoint = start_date + timedelta(days=int((end_date - start_date).days * (j + 1) / points_per_exp))
                    berufserfahrung = (timepoint - self.parse_date(experiences[-1]['startDate'])).days
                    anzahl_wechsel = sum(1 for e in experiences if e['endDate'] != "Present" and self.parse_date(e['endDate']) <= timepoint)
                    anzahl_jobs = sum(1 for e in experiences if self.parse_date(e['startDate']) <= timepoint)
                    dauer_liste = []
                    for e in experiences:
                        s = self.parse_date(e['startDate'])
                        e_date = self.parse_date(e['endDate'])
                        if s < e_date and e_date <= timepoint:
                            dauer_liste.append((e_date - s).days)
                    durchschnittsdauer = sum(dauer_liste) / len(dauer_liste) if dauer_liste else 0
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
        return time_points, experiences

    def prepare_prediction_dataframe(self, time_points, min_required_points=24):
        if not time_points:
            print("Keine gültigen Zeitpunkte gefunden")
            return None
        df = pd.DataFrame(time_points)
        if len(df) < min_required_points:
            print(f"Warnung: Zu wenige Zeitpunkte ({len(df)}). Mindestens {min_required_points} erforderlich.")
            additional_points_needed = min_required_points - len(df)
            if additional_points_needed > 0:
                last_points = df.iloc[-additional_points_needed:].copy()
                last_points['time_idx'] = range(len(df), len(df) + additional_points_needed)
                df = pd.concat([df, last_points])
        df["time_idx"] = df["time_idx"].astype(int)
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
        return df 