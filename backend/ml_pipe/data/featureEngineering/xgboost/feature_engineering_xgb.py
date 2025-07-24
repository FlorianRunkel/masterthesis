import torch
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import json
import os
from rapidfuzz import process, fuzz

'''
Feature Engineering for XGBoost
'''
class FeatureEngineering:

    '''
    Initialize FeatureEngineering
    '''
    def __init__(self, use_llm: bool = False):
        script_dir = os.path.dirname(__file__)
        json_path = os.path.join(script_dir, '..', 'position_level.json')

        with open(json_path, "r", encoding="utf-8") as f:
            position_list = json.load(f)
        self.position_map = {entry["position"]: (entry["level"], entry["branche"], entry["durchschnittszeit_tage"]) for entry in position_list}
        self.all_positions = list(self.position_map.keys())

    '''
    Fuzzy match position
    '''
    def find_best_match(self, pos, threshold=50):
        pos_clean = pos.lower().strip()
        if pos_clean in self.position_map:
            level, branche, durchschnittszeit_tage = self.position_map[pos_clean]
            return level, branche, durchschnittszeit_tage

        match, score, _ = process.extractOne(pos_clean, self.all_positions, scorer=fuzz.ratio)
        if score >= threshold:
            level, branche, durchschnittszeit_tage = self.position_map[match]
            return level, branche, durchschnittszeit_tage

        raise ValueError(f"Position '{pos}' could not be mapped (Score: {score}).")

    '''
    Months between
    '''
    def months_between(self, start, end):
        delta = relativedelta(end, start)
        return delta.years * 12 + delta.months

    '''
    Parse date
    '''
    def parse_date(self, date_str):
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

    '''
    Extract features from single user
    '''
    def extract_features_from_single_user(self, user_doc: dict) -> np.ndarray:
        experiences = user_doc.get("experiences", user_doc.get("career_history", []))

        if not experiences:
            print("No experiences found")
            return np.zeros((1, 3), dtype=np.float32)  # Default-Features zurückgeben

        valid_experiences = []
        for exp in experiences:
            if exp.get("position") and exp.get("startDate"):
                valid_experiences.append(exp)

        if not valid_experiences:
            print("No valid experiences with position and start date found")
            return np.zeros((1, 3), dtype=np.float32)

        valid_experiences = sorted(
            valid_experiences, 
            key=lambda x: self.parse_date(x.get("startDate", "")),
            reverse=True
        )

        current_position = valid_experiences[0]

        start_date = self.parse_date(current_position.get("startDate", ""))
        end_date = datetime.now()

        if not start_date:
            print("No valid start date found")
            return np.zeros((1, 3), dtype=np.float32)

        duration_months = self.months_between(start_date, end_date)

        position_title = current_position.get("position", "").strip()
        level, branche, durchschnittszeit_tage = self.find_best_match(position_title)

        expected = self.expected_duration.get(
            (branche, level),
            self.expected_duration.get((0, level), 24)  # Default: 2 Years
        )

        normalized_duration = min(duration_months / expected, 1.0)


        features = np.array([
            normalized_duration,
            float(level),
            float(branche)
        ], dtype=np.float32)

        return features.reshape(1, -1)

    '''
    Extract features and labels for training
    '''
    def extract_features_and_labels_for_training(self, documents):

        all_sequences = []
        all_labels = []

        for doc in documents:
            history = doc.get("career_history", [])
            if len(history) < 2:
                continue

            valid_history = []
            for job in history:
                start_date = self.parse_date(job.get("startDate", ""))
                if start_date is not None:
                    valid_history.append((start_date, job))

            if len(valid_history) < 2:  # min 2 valid positions
                continue

            valid_history.sort(key=lambda x: x[0])
            history = [job for _, job in valid_history]

            position_features = []
            for i, job in enumerate(history):
                start_date = self.parse_date(job.get("startDate", ""))
                end_date = self.parse_date(job.get("endDate", ""))
                if not end_date:
                    end_date = datetime.now()

                if not start_date or not end_date:
                    continue

                duration_months = self.months_between(start_date, end_date)
                level, branche, durchschnittszeit_tage = self.find_best_match(job.get("position", "").strip())

                expected = self.expected_duration.get(
                    (branche, level),
                    self.expected_duration.get((0, level), 24)
                )

                normalized_duration = min(duration_months / expected, 1.0)

                position_feature = [
                    normalized_duration,
                    float(level),
                    float(branche)
                ]

                position_features.append(position_feature)

                if i < len(history) - 1:
                    next_job = history[i + 1]
                    next_level, next_branche, _ = self.find_best_match(next_job.get("position", "").strip())

                    changed_branch = branche != next_branche
                    stayed_long = normalized_duration >= 0.8

                    label = 1.0 if (changed_branch or stayed_long) else 0.0

                    if len(position_features) > 0:
                        all_sequences.append(position_features[:i+1])
                        all_labels.append([label])

        if not all_sequences:
            return torch.empty(0), torch.empty(0)

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

    '''
    Extract features for xgboost
    '''
    def extract_features_for_xgboost(self, user_doc: dict) -> np.ndarray:
        experiences = user_doc.get("experiences", user_doc.get("career_history", []))

        if not experiences:
            print("No experiences found")
            return np.zeros(51, dtype=np.float32)

        valid_experiences = []
        for exp in experiences:
            if exp.get("position") and exp.get("startDate"):
                start_date = self.parse_date(exp.get("startDate", ""))
                if start_date is not None:
                    valid_experiences.append((start_date, exp))

        if not valid_experiences:
            print("No valid experiences with position and start date found")
            return np.zeros(51, dtype=np.float32)

        valid_experiences.sort(key=lambda x: x[0], reverse=True)
        features = np.zeros(51, dtype=np.float32)

        for i, (_, exp) in enumerate(valid_experiences[:17]): # max 17 positions
            start_date = self.parse_date(exp.get("startDate", ""))
            end_date = self.parse_date(exp.get("endDate", ""))

            if not end_date:
                end_date = datetime.now()
            if not start_date:
                continue

            try:
                duration_months = self.months_between(start_date, end_date)

                position_title = exp.get("position", "").strip()
                level, branche, durchschnittszeit_tage = self.find_best_match(position_title)

                expected = self.expected_duration.get(
                    (branche, level),
                    self.expected_duration.get((0, level), 24)
                )

                normalized_duration = min(duration_months / expected, 1.0)

                base_idx = i * 3
                features[base_idx] = normalized_duration
                features[base_idx + 1] = float(level)
                features[base_idx + 2] = float(branche)
            except Exception as e:
                print(f"Fehler bei der Verarbeitung von Position {i}: {str(e)}")
                continue

        return features

    '''
    Extract features for prediction
    '''
    def extract_features_for_prediction(self, user_doc: dict) -> np.ndarray:
        experiences = user_doc.get("experiences", user_doc.get("career_history", []))

        if not experiences:
            print("No experiences found")
            return np.zeros(51, dtype=np.float32) # 17 × 3 = 51 Features

        valid_experiences = []
        for exp in experiences:
            if exp.get("position") and exp.get("startDate"):
                start_date = self.parse_date(exp.get("startDate", ""))
                if start_date is not None:
                    valid_experiences.append((start_date, exp))

        if not valid_experiences:
            print("No valid experiences with position and start date found")
            return np.zeros(51, dtype=np.float32)

        valid_experiences.sort(key=lambda x: x[0], reverse=True)
        features = np.zeros(51, dtype=np.float32)

        for i, (_, exp) in enumerate(valid_experiences[:17]):
            start_date = self.parse_date(exp.get("startDate", ""))
            end_date = self.parse_date(exp.get("endDate", ""))
            if not end_date:
                end_date = datetime.now()

            if not start_date:
                continue

            try:
                duration_months = self.months_between(start_date, end_date)

                position_title = exp.get("position", "").strip()
                level, branche, durchschnittszeit_tage = self.find_best_match(position_title)

                expected = self.expected_duration.get(
                    (branche, level),
                    self.expected_duration.get((0, level), 24)
                )

                normalized_duration = min(duration_months / expected, 1.0)

                base_idx = i * 3
                features[base_idx] = normalized_duration
                features[base_idx + 1] = float(level)
                features[base_idx + 2] = float(branche)
            except Exception as e:
                print(f"Fehler bei der Verarbeitung von Position {i}: {str(e)}")
                continue

        return features.reshape(1, -1)  # Reshape für Vorhersage (1, 51)