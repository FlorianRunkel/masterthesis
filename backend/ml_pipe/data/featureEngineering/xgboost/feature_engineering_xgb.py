import torch
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import json
from typing import Tuple
import os
from rapidfuzz import process, fuzz

class FeatureEngineering:
    
    def __init__(self, use_llm: bool = False):
        # Lade die Position-Level-Zuordnungen relativ zum Skriptpfad
        script_dir = os.path.dirname(__file__)
        json_path = os.path.join(script_dir, '..', 'position_level.json')
        # Lade die Position-Level-Zuordnungen als Dict: {"position": (level, branche, durchschnittszeit_tage)}
        with open(json_path, "r", encoding="utf-8") as f:
            position_list = json.load(f)
        self.position_map = {entry["position"]: (entry["level"], entry["branche"], entry["durchschnittszeit_tage"]) for entry in position_list}
        self.all_positions = list(self.position_map.keys())

    def find_best_match(self, pos, threshold=50):
        pos_clean = pos.lower().strip()
        # Exaktes Matching
        if pos_clean in self.position_map:
            level, branche, durchschnittszeit_tage = self.position_map[pos_clean]
            return level, branche, durchschnittszeit_tage
        # Fuzzy-Matching
        match, score, _ = process.extractOne(pos_clean, self.all_positions, scorer=fuzz.ratio)
        if score >= threshold:
            level, branche, durchschnittszeit_tage = self.position_map[match]
            return level, branche, durchschnittszeit_tage
        # Kein guter Treffer
        raise ValueError(f"Position '{pos}' konnte nicht sicher gemappt werden (Score: {score}).")

    def months_between(self, start, end):
        """Berechnet die Anzahl der Monate zwischen zwei Datumswerten"""
        delta = relativedelta(end, start)
        return delta.years * 12 + delta.months

    def parse_date_safe(self, date_str):
        """
        Versucht ein Datum aus verschiedenen Formaten zu parsen.
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

    def extract_features_from_single_user(self, user_doc: dict) -> np.ndarray:
        # Versuche beide möglichen Keys
        experiences = user_doc.get("experiences", user_doc.get("career_history", []))
        
        if not experiences:
            print("Keine Erfahrungen gefunden")
            return np.zeros((1, 3), dtype=np.float32)  # Default-Features zurückgeben

        # Filtere ungültige Einträge
        valid_experiences = []
        for exp in experiences:
            if exp.get("position") and exp.get("startDate"):
                valid_experiences.append(exp)

        if not valid_experiences:
            print("Keine gültigen Erfahrungen mit Position und Startdatum gefunden")
            return np.zeros((1, 3), dtype=np.float32)

        # Sortiere Erfahrungen nach Datum
        valid_experiences = sorted(
            valid_experiences, 
            key=lambda x: self.parse_date_safe(x.get("startDate", "")),
            reverse=True  # Neueste zuerst
        )
        
        current_position = valid_experiences[0]  # Aktuellste Position
        
        # Features für aktuelle Position
        start_date = self.parse_date_safe(current_position.get("startDate", ""))
        end_date = datetime.now()  # Aktuelle Position geht bis heute
        
        if not start_date:
            print("Kein gültiges Startdatum gefunden")
            return np.zeros((1, 3), dtype=np.float32)

        # Berechne Features für aktuelle Position
        duration_months = self.months_between(start_date, end_date)
        
        # Position Features
        position_title = current_position.get("position", "").strip()
        level, branche, durchschnittszeit_tage = self.find_best_match(position_title)
        
        # Hole erwartete Verweildauer für Position und Branche
        expected = self.expected_duration.get(
            (branche, level),
            self.expected_duration.get((0, level), 24)  # Default: 2 Jahre
        )
        
        # Normalisiere Dauer basierend auf erwarteter Dauer
        normalized_duration = min(duration_months / expected, 1.0)
        
        # Erstelle Feature-Vektor
        features = np.array([
            normalized_duration,  # Normalisierte Dauer (0-1, branchenspezifisch)
            float(level),        # Level (1-8)
            float(branche)       # Branche (0-3)
        ], dtype=np.float32)

        # Reshape für GRU Modell (1, 3)
        return features.reshape(1, -1)

    def extract_features_and_labels_for_training(self, documents):

        all_sequences = []
        all_labels = []

        for doc in documents:
            history = doc.get("career_history", [])
            if len(history) < 2:  # Mindestens 2 Positionen nötig
                continue
            
            # Sortiere nach Datum, ignoriere Einträge ohne gültiges Startdatum
            valid_history = []
            for job in history:
                start_date = self.parse_date_safe(job.get("startDate", ""))
                if start_date is not None:
                    valid_history.append((start_date, job))
            
            if len(valid_history) < 2:  # Mindestens 2 gültige Positionen nötig
                continue
            
            # Sortiere nach Datum
            valid_history.sort(key=lambda x: x[0])
            history = [job for _, job in valid_history]
            
            # Extrahiere Features für jede Position
            position_features = []
            
            for i, job in enumerate(history):
                # Parse Daten
                start_date = self.parse_date_safe(job.get("startDate", ""))
                end_date = self.parse_date_safe(job.get("endDate", ""))
                if not end_date:  # Wenn kein Enddatum, dann ist es die aktuelle Position
                    end_date = datetime.now()
                
                if not start_date or not end_date:
                    continue

                # Berechne Basisdaten
                duration_months = self.months_between(start_date, end_date)
                level, branche, durchschnittszeit_tage = self.find_best_match(job.get("position", "").strip())
                
                # Hole erwartete Verweildauer
                expected = self.expected_duration.get(
                    (branche, level),
                    self.expected_duration.get((0, level), 24)
                )
                
                # Normalisiere Dauer
                normalized_duration = min(duration_months / expected, 1.0)
                
                # Erstelle Feature-Vektor für diese Position
                position_feature = [
                    normalized_duration,  # Relative Dauer zur erwarteten Dauer
                    float(level),        # Position Level
                    float(branche)       # Branche
                ]
                
                position_features.append(position_feature)
                
                # Wenn es nicht die letzte Position ist, erstelle Label
                if i < len(history) - 1:
                    # Prüfe ob die nächste Position ein Wechsel war
                    next_job = history[i + 1]
                    next_level, next_branche, _ = self.find_best_match(next_job.get("position", "").strip())
                    
                    # Label ist 1 wenn:
                    # - Person die Branche gewechselt hat ODER
                    # - Person länger als erwartet in der Position war
                    changed_branch = branche != next_branche
                    stayed_long = normalized_duration >= 0.8  # 80% der erwarteten Zeit
                    
                    label = 1.0 if (changed_branch or stayed_long) else 0.0
                    
                    # Füge Sequenz und Label hinzu
                    if len(position_features) > 0:
                        all_sequences.append(position_features[:i+1])  # Alle Positionen bis zur aktuellen
                        all_labels.append([label])

        if not all_sequences:
            return torch.empty(0), torch.empty(0)

        # Padding für Sequenzen auf gleiche Länge
        max_len = max(len(seq) for seq in all_sequences)
        padded_sequences = []
        
        for seq in all_sequences:
            if len(seq) < max_len:
                padding = [[0.0, 0.0, 0.0] for _ in range(max_len - len(seq))]
                padded_seq = seq + padding
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)

        return (
            torch.tensor(padded_sequences, dtype=torch.float32),
            torch.tensor(all_labels, dtype=torch.float32)
        )

    def extract_features_for_xgboost(self, user_doc: dict) -> np.ndarray:
        """
        Extrahiert Features für das XGBoost-Modell.
        Gibt eine flache Feature-Vektor mit 51 Features zurück (17 Schritte × 3 Features).
        """
        # Versuche beide möglichen Keys
        experiences = user_doc.get("experiences", user_doc.get("career_history", []))
        
        if not experiences:
            print("Keine Erfahrungen gefunden")
            return np.zeros(51, dtype=np.float32)  # 17 × 3 = 51 Features

        # Filtere ungültige Einträge und konvertiere zu Tupeln mit Datum
        valid_experiences = []
        for exp in experiences:
            if exp.get("position") and exp.get("startDate"):
                start_date = self.parse_date_safe(exp.get("startDate", ""))
                if start_date is not None:  # Nur Einträge mit gültigem Startdatum
                    valid_experiences.append((start_date, exp))

        if not valid_experiences:
            print("Keine gültigen Erfahrungen mit Position und Startdatum gefunden")
            return np.zeros(51, dtype=np.float32)

        # Sortiere Erfahrungen nach Datum
        valid_experiences.sort(key=lambda x: x[0], reverse=True)  # Neueste zuerst
        
        # Erstelle Feature-Array mit 51 Features
        features = np.zeros(51, dtype=np.float32)
        
        # Fülle die ersten Positionen mit den tatsächlichen Daten
        for i, (_, exp) in enumerate(valid_experiences[:17]):  # Maximal 17 Positionen
            start_date = self.parse_date_safe(exp.get("startDate", ""))
            end_date = self.parse_date_safe(exp.get("endDate", ""))
            
            # Wenn kein Enddatum, verwende aktuelles Datum
            if not end_date:
                end_date = datetime.now()
            
            # Wenn kein gültiges Startdatum, überspringe diese Position
            if not start_date:
                continue

            try:
                # Berechne Features für diese Position
                duration_months = self.months_between(start_date, end_date)
                
                # Position Features
                position_title = exp.get("position", "").strip()
                level, branche, durchschnittszeit_tage = self.find_best_match(position_title)
                
                # Hole erwartete Verweildauer für Position und Branche
                expected = self.expected_duration.get(
                    (branche, level),
                    self.expected_duration.get((0, level), 24)
                )
                
                # Normalisiere Dauer basierend auf erwarteter Dauer
                normalized_duration = min(duration_months / expected, 1.0)
                
                # Speichere Features an der richtigen Position im Array
                base_idx = i * 3
                features[base_idx] = normalized_duration    # Normalisierte Dauer
                features[base_idx + 1] = float(level)      # Level
                features[base_idx + 2] = float(branche)  # Branche
            except Exception as e:
                print(f"Fehler bei der Verarbeitung von Position {i}: {str(e)}")
                continue
        
        return features

    def extract_features_for_prediction(self, user_doc: dict) -> np.ndarray:
        """
        Extrahiert Features für die Vorhersage mit XGBoost.
        Gibt einen flachen Feature-Vektor mit 51 Features zurück (17 Schritte × 3 Features).
        """
        # Versuche beide möglichen Keys
        experiences = user_doc.get("experiences", user_doc.get("career_history", []))
        
        if not experiences:
            print("Keine Erfahrungen gefunden")
            return np.zeros(51, dtype=np.float32)  # 17 × 3 = 51 Features

        # Filtere ungültige Einträge und konvertiere zu Tupeln mit Datum
        valid_experiences = []
        for exp in experiences:
            if exp.get("position") and exp.get("startDate"):
                start_date = self.parse_date_safe(exp.get("startDate", ""))
                if start_date is not None:  # Nur Einträge mit gültigem Startdatum
                    valid_experiences.append((start_date, exp))

        if not valid_experiences:
            print("Keine gültigen Erfahrungen mit Position und Startdatum gefunden")
            return np.zeros(51, dtype=np.float32)

        # Sortiere Erfahrungen nach Datum
        valid_experiences.sort(key=lambda x: x[0], reverse=True)  # Neueste zuerst
        
        # Erstelle Feature-Array mit 51 Features
        features = np.zeros(51, dtype=np.float32)
        
        # Fülle die ersten Positionen mit den tatsächlichen Daten
        for i, (_, exp) in enumerate(valid_experiences[:17]):  # Maximal 17 Positionen
            start_date = self.parse_date_safe(exp.get("startDate", ""))
            end_date = self.parse_date_safe(exp.get("endDate", ""))
            if not end_date:
                end_date = datetime.now()
            
            if not start_date:
                continue

            try:
                # Berechne Features für diese Position
                duration_months = self.months_between(start_date, end_date)
                
                # Position Features
                position_title = exp.get("position", "").strip()
                level, branche, durchschnittszeit_tage = self.find_best_match(position_title)
                
                # Hole erwartete Verweildauer für Position und Branche
                expected = self.expected_duration.get(
                    (branche, level),
                    self.expected_duration.get((0, level), 24)
                )
                
                # Normalisiere Dauer basierend auf erwarteter Dauer
                normalized_duration = min(duration_months / expected, 1.0)
                
                # Speichere Features an der richtigen Position im Array
                base_idx = i * 3
                features[base_idx] = normalized_duration    # Normalisierte Dauer
                features[base_idx + 1] = float(level)      # Level
                features[base_idx + 2] = float(branche)  # Branche
            except Exception as e:
                print(f"Fehler bei der Verarbeitung von Position {i}: {str(e)}")
                continue
        
        return features.reshape(1, -1)  # Reshape für Vorhersage (1, 51)