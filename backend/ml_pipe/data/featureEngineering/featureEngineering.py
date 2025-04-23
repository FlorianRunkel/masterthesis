import torch
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from collections import defaultdict
import json
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import List, Tuple, Dict
from difflib import SequenceMatcher
import re
import os
import pandas as pd

class featureEngineering:
    def __init__(self):
        # Bestimme den absoluten Pfad zur JSON-Datei relativ zum aktuellen Skript
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "position_level.json")
        
        # Lade die Position-Level-Zuordnungen
        with open(json_path, "r", encoding="utf-8") as f:
            self.position_levels = json.load(f)

        self.branch_mapping = self._create_branch_mapping()
        
        # Erwartete Verweildauer pro Level und Branche (in Monaten)
        self.expected_duration = {
            # Sales
            ("sales", 1): 12,     # Junior Sales: 1 Jahr
            ("sales", 2): 24,     # Regular Sales: 2 Jahre
            ("sales", 3): 36,     # Senior Sales: 3 Jahre
            ("sales", 4): 48,     # Lead Sales: 4 Jahre
            
            # Engineering
            ("engineering", 1): 18,    # Junior Dev: 1.5 Jahre
            ("engineering", 2): 30,    # Regular Dev: 2.5 Jahre
            ("engineering", 3): 42,    # Senior Dev: 3.5 Jahre
            ("engineering", 4): 54,    # Lead Dev: 4.5 Jahre
            
            # Marketing
            ("marketing", 1): 15,      # Junior Marketing: 1.25 Jahre
            ("marketing", 2): 27,      # Regular Marketing: 2.25 Jahre
            ("marketing", 3): 39,      # Senior Marketing: 3.25 Jahre
            ("marketing", 4): 51,      # Lead Marketing: 4.25 Jahre
            
            # Default für andere Branchen
            ("default", 1): 15,        # Junior: 1.25 Jahre
            ("default", 2): 27,        # Regular: 2.25 Jahre
            ("default", 3): 39,        # Senior: 3.25 Jahre
            ("default", 4): 51,        # Lead: 4.25 Jahre
        }
        
        # Erstelle eine Liste aller Positionen mit ihren Eigenschaften
        self.all_positions = []
        for entry in self.position_levels:
            self.all_positions.append({
                'position': entry['position'].lower(),
                'level': entry['level'],
                'branche': entry['branche'].lower()
            })
        
        # Sortiere Positionen nach Länge (längere zuerst)
        self.all_positions.sort(key=lambda x: len(x['position']), reverse=True)

    def _create_branch_mapping(self) -> Dict[str, int]:
        return {
            "marketing": 1,
            "engineering": 2,
            "finance": 3,
            "product": 4,
            "it": 5,
            "consulting": 6,
            "hr": 7,
            "daten & analyse": 8,
            "customer success": 9,
            "management": 10,
            "sales": 12,
        }

    def build_position_mapping(self, entries):
        mapping = {}
        for entry in entries:
            branche = entry["branche"].lower()
            title = entry["position"].lower()
            level = entry["level"]

            if branche not in mapping:
                mapping[branche] = {}
            mapping[branche][title] = level
        return mapping

    def get_branch_code(self, branche: str) -> int:
        return self.branch_mapping.get(branche.lower(), 0)

    def similarity(self, a: str, b: str) -> float:
        """Berechnet die Ähnlichkeit zwischen zwei Strings"""
        return SequenceMatcher(None, a, b).ratio()

    def find_best_match(self, title: str) -> Tuple[int, str]:
        """Findet die beste Übereinstimmung für einen Positionstitel"""
        if not title:
            return 2, "other"  # Default: Mid-Level, Other

        title = title.lower()
        best_match = None
        best_score = 0.7

        for pos in self.all_positions:
            if pos['position'] in title:
                return pos['level'], pos['branche']

        for pos in self.all_positions:
            score = self.similarity(title, pos['position'])
            if score > best_score:
                best_score = score
                best_match = pos

        if best_match:
            return best_match['level'], best_match['branche']
        
        return 2, self._determine_branch(title)

    def _determine_branch(self, title: str) -> str:
        """Bestimmt die Branche basierend auf Keywords im Titel"""
        title = title.lower()
        
        if any(kw in title for kw in ["sales", "account", "business"]):
            return "sales"
        elif any(kw in title for kw in ["engineer", "developer", "programming"]):
            return "engineering"
        elif any(kw in title for kw in ["marketing", "brand", "content"]):
            return "marketing"
        elif any(kw in title for kw in ["finance", "accounting", "controlling"]):
            return "finance"
        elif any(kw in title for kw in ["product", "project"]):
            return "product"
        elif any(kw in title for kw in ["hr", "human resources", "recruiting"]):
            return "hr"
        elif any(kw in title for kw in ["data", "analytics", "analysis"]):
            return "daten & analyse"
        
        return "other"

    def months_between(self, start, end):
        delta = relativedelta(end, start)
        return delta.years * 12 + delta.months

    def parse_date_safe(self, date_str):
        """
        Versucht ein Datum aus verschiedenen Formaten zu parsen.
        Handhabt auch None, leere Strings und Dictionary-Formate.
        """
        if date_str is None:
            return None
        
        if isinstance(date_str, dict):
            # Wenn es ein Dictionary ist, versuche die relevanten Felder zu extrahieren
            if 'year' in date_str and 'month' in date_str:
                try:
                    return datetime(int(date_str['year']), int(date_str['month']), 1)
                except (ValueError, TypeError):
                    return None
            return None
        
        if not isinstance(date_str, str) or not date_str.strip():
            return None
        
        # Entferne führende/folgende Leerzeichen
        date_str = date_str.strip()
        
        # Liste der möglichen Datumsformate
        date_formats = [
            '%Y-%m-%d',    # ISO Format
            '%d.%m.%Y',    # Deutsches Format
            '%m/%d/%Y',    # Amerikanisches Format
            '%Y-%m',       # Nur Jahr und Monat
            '%Y'           # Nur Jahr
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
            
        # Wenn kein Format passt, versuche das Jahr zu extrahieren
        year_match = re.search(r'\d{4}', date_str)
        if year_match:
            try:
                return datetime(int(year_match.group()), 1, 1)
            except ValueError:
                return None
            
        return None

    def extract_features_from_single_user(self, user_doc: dict) -> np.ndarray:
        """Extrahiert Features für die aktuelle Position"""
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
        level, branche = self.find_best_match(position_title)
        branche_code = self.get_branch_code(branche)
        
        # Hole erwartete Verweildauer für Position und Branche
        expected = self.expected_duration.get(
            (branche.lower(), level),
            self.expected_duration.get(("default", level), 24)  # Default: 2 Jahre
        )
        
        # Normalisiere Dauer basierend auf erwarteter Dauer
        normalized_duration = min(duration_months / expected, 1.0)
        
        # Erstelle Feature-Vektor
        features = np.array([
            normalized_duration,  # Normalisierte Dauer (0-1, branchenspezifisch)
            float(level),        # Level (1-4)
            float(branche_code)  # Branche (1-12)
        ], dtype=np.float32)
        
        print(f"""
        Feature-Extraktion Details:
        --------------------------
        Aktuelle Position: {position_title}
        Start: {start_date.strftime('%Y-%m-%d')}
        Dauer: {duration_months:.1f} Monate
        Normalisierte Dauer: {normalized_duration:.2f}
        Level: {level}
        Branche: {branche} (Code: {branche_code})
        Erwartete Dauer: {expected} Monate
        """)
        
        # Reshape für GRU Modell (1, 3)
        return features.reshape(1, -1)

    def extract_features_and_labels_for_training(self, documents):
        """
        Extrahiert Features und Labels für das Training aus einer Sammlung von Karriereverläufen.
        Jede Position wird als potentieller Wechselpunkt betrachtet.
        
        Args:
            documents: Liste von Dokumenten mit Karriereverläufen
            
        Returns:
            features: Liste von Feature-Sequenzen [duration, level, branche]
            labels: Liste von Labels (1 = Person hat gewechselt, 0 = Person ist geblieben)
        """
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
                level, branche = self.find_best_match(job.get("position", "").strip())
                branche_code = self.get_branch_code(branche)
                
                # Hole erwartete Verweildauer
                expected = self.expected_duration.get(
                    (branche.lower(), level),
                    self.expected_duration.get(("default", level), 24)
                )
                
                # Normalisiere Dauer
                normalized_duration = min(duration_months / expected, 1.0)
                
                # Erstelle Feature-Vektor für diese Position
                position_feature = [
                    normalized_duration,  # Relative Dauer zur erwarteten Dauer
                    float(level),        # Position Level
                    float(branche_code)  # Branche
                ]
                
                position_features.append(position_feature)
                
                # Wenn es nicht die letzte Position ist, erstelle Label
                if i < len(history) - 1:
                    # Prüfe ob die nächste Position ein Wechsel war
                    next_job = history[i + 1]
                    next_level, next_branche = self.find_best_match(next_job.get("position", "").strip())
                    
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
                        '''
                        print(f"""
                        Position {i+1}: {job.get('position')}
                        Dauer: {duration_months:.1f} Monate (normalisiert: {normalized_duration:.2f})
                        Level: {level}, Branche: {branche}
                        Nächste Position: {next_job.get('position')}
                        Branchenwechsel: {changed_branch}, Lange geblieben: {stayed_long}
                        Label: {label}
                        """)
                        '''

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
                level, branche = self.find_best_match(position_title)
                branche_code = self.get_branch_code(branche)
                
                # Hole erwartete Verweildauer für Position und Branche
                expected = self.expected_duration.get(
                    (branche.lower(), level),
                    self.expected_duration.get(("default", level), 24)  # Default: 2 Jahre
                )
                
                # Normalisiere Dauer basierend auf erwarteter Dauer
                normalized_duration = min(duration_months / expected, 1.0)
                
                # Speichere Features an der richtigen Position im Array
                base_idx = i * 3
                features[base_idx] = normalized_duration    # Normalisierte Dauer
                features[base_idx + 1] = float(level)      # Level
                features[base_idx + 2] = float(branche_code)  # Branche
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
                level, branche = self.find_best_match(position_title)
                branche_code = self.get_branch_code(branche)
                
                # Hole erwartete Verweildauer für Position und Branche
                expected = self.expected_duration.get(
                    (branche.lower(), level),
                    self.expected_duration.get(("default", level), 24)  # Default: 2 Jahre
                )
                
                # Normalisiere Dauer basierend auf erwarteter Dauer
                normalized_duration = min(duration_months / expected, 1.0)
                
                # Speichere Features an der richtigen Position im Array
                base_idx = i * 3
                features[base_idx] = normalized_duration    # Normalisierte Dauer
                features[base_idx + 1] = float(level)      # Level
                features[base_idx + 2] = float(branche_code)  # Branche
            except Exception as e:
                print(f"Fehler bei der Verarbeitung von Position {i}: {str(e)}")
                continue
        
        return features.reshape(1, -1)  # Reshape für Vorhersage (1, 51)