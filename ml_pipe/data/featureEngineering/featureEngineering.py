import torch
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from collections import defaultdict
import json

class featureEngineering:
    def __init__(self):
        json_path = "/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/ml_pipe/data/featureEngineering/position_level.json"
        with open(json_path, "r", encoding="utf-8") as f:
            raw_levels = json.load(f)

        self.position_levels = self.build_position_mapping(raw_levels)

        self.branch_mapping = {
            "marketing": 1,
            "engineering": 2,
            "finance": 3,
            "product": 4,
            "it": 5,
            "consulting": 6,
            "hr": 7,
            "gastronomie": 8,
            "daten & analyse": 9,
            "customer success": 10,
            "management": 11,
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

    def get_position_level(self, title: str) -> tuple[int, int]:
        if not title:
            return 2, 0

        title = title.lower()
        for branche, titles in self.position_levels.items():
            for known_title in titles:
                if known_title in title:
                    level = titles[known_title]
                    branche_code = self.get_branch_code(branche)
                    return level, branche_code
        return 2, 0

    def months_between(self, start, end):
        delta = relativedelta(end, start)
        return delta.years * 12 + delta.months

    def parse_date_safe(self, date_str):
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d.%m.%Y"):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None

    def compute_avg_durations_per_role(self, documents) -> dict:
        role_durations = defaultdict(list)
        for doc in documents:
            for job in doc.get("career_history", []):
                title = job.get("position", "").lower()
                duration_str = job.get("duration")
                if not title or not duration_str:
                    continue
                try:
                    start_str, end_str = duration_str.split(" - ")
                    start = int(start_str[-4:])
                    end = int(end_str[-4:]) if "present" not in end_str.lower() else 2025
                    duration_months = (end - start) * 12
                    role_durations[title].append(duration_months)
                except:
                    continue
        return {
            title: sum(durations) / len(durations)
            for title, durations in role_durations.items() if durations
        }

    def extract_features_and_labels(self, documents):
        feature_seqs = []
        labels = []

        avg_role_duration = self.compute_avg_durations_per_role(documents)

        for doc in documents:
            history = doc.get("career_history", [])
            if len(history) < 2:
                continue

            features = []
            durations = []

            for job in history:
                duration_str = job.get("duration")
                title = job.get("position", "")
                if duration_str:
                    try:
                        start_str, end_str = duration_str.split(" - ")
                        start = int(start_str[-4:])
                        end = int(end_str[-4:]) if "present" not in end_str.lower() else 2025
                        duration_months = (end - start) * 12
                        level, branche_code = self.get_position_level(title)
                        features.append([duration_months, level, branche_code])
                        durations.append(duration_months)
                    except:
                        continue

            if len(features) >= 2:
                seq = torch.tensor(features[:-1], dtype=torch.float32)
                last_title = history[-1].get("position", "").lower()
                last_duration = durations[-1]
                avg_duration = avg_role_duration.get(last_title)

                if avg_duration:
                    label_value = 1.0 if last_duration >= avg_duration else 0.0
                    label = torch.tensor([label_value])
                    feature_seqs.append(seq)
                    labels.append(label)

        if not feature_seqs:
            return torch.empty(0), torch.empty(0)

        max_len = max(len(seq) for seq in feature_seqs)
        input_size = feature_seqs[0].shape[-1]

        padded = [torch.cat([seq, torch.zeros(max_len - len(seq), input_size)], dim=0) for seq in feature_seqs]

        return torch.stack(padded), torch.stack(labels)

    def extract_features_from_single_user(self, user_doc: dict):
        expected_features = 51  # z. B. 17 Jobs × 3 Features
        experiences = user_doc.get("career_history", [])
        feature_seq = []

        for job in experiences:
            try:
                start = self.parse_date_safe(job["startDate"])
                end = self.parse_date_safe(job["endDate"]) if job["endDate"].lower() != "present" else datetime.now()

                if not start or not end:
                    print(f"Ungültiges Datum in Job: {job}")
                    continue

                duration = self.months_between(start, end)
                level, branche_code = self.get_position_level(job.get("position", ""))
                feature_seq.append([duration, level, branche_code])
            except Exception as e:
                print(f"Fehler bei {job}: {e}")
                continue

        if not feature_seq:
            return np.zeros((1, expected_features), dtype=np.float32)

        flat = np.array(feature_seq, dtype=np.float32).flatten().reshape(1, -1)
        current_len = flat.shape[1]

        if current_len < expected_features:
            pad = np.zeros((1, expected_features - current_len), dtype=np.float32)
            flat = np.hstack([flat, pad])
        elif current_len > expected_features:
            flat = flat[:, :expected_features]

        return flat