import torch
from collections import defaultdict
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

"""
Diese Klasse enthält Methoden zur Extraktion, Transformation und Aufbereitung von Karrieredaten für maschinelles Lernen.
"""
class featureEngineering:
    """
    Gibt die Hierarchie-Stufe einer Position im Sales-Bereich zurück (0 = niedrig, 6 = sehr senior).
    """
    def get_position_level(self, title: str) -> int:

        if not title:
            return 2

        title = title.lower()

        if "intern" in title or "working student" in title or "werkstudent" in title:
            return 0
        elif any(role in title for role in ["sales development representative", "business development representative", "inside sales", "sdr", "bdr", "junior"]):
            return 1
        elif any(role in title for role in ["account executive", "sales representative", "sales rep", "mid-market", "field sales"]):
            return 2
        elif "senior" in title:
            return 3
        elif "lead" in title or "team lead" in title:
            return 4
        elif any(role in title for role in ["manager", "head of sales", "sales manager"]):
            return 5
        elif any(role in title for role in ["vp", "director", "chief revenue officer", "cro"]):
            return 6
        else:
            return 2

    """
    Berechnet die durchschnittliche Verweildauer pro Jobtitel über alle Kandidaten hinweg.
    """
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
        """
        Erzeugt Sequenzen von Features (Verweildauer + Level) und dazugehörige Labels:
        1.0 = Kandidat ist länger als durchschnittlich in aktueller Position → evtl. wechselbereit
        """
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
                title = job.get("position", None)
                if duration_str:
                    try:
                        start_str, end_str = duration_str.split(" - ")
                        start = int(start_str[-4:])
                        end = int(end_str[-4:]) if "present" not in end_str.lower() else 2025
                        duration_months = (end - start) * 12
                        level = self.get_position_level(title)
                        features.append([duration_months, level])
                        durations.append(duration_months)
                    except:
                        continue

            if len(features) >= 2:
                seq = torch.tensor(features[:-1], dtype=torch.float32)
                last_job = history[-1]
                last_title = last_job.get("position", "").lower()
                last_duration = durations[-1]
                avg_duration = avg_role_duration.get(last_title)

                if avg_duration:
                    label_value = 1.0 if last_duration >= avg_duration else 0.0
                    label = torch.tensor([label_value])
                    feature_seqs.append(seq)
                    labels.append(label)

        if not feature_seqs:
            return torch.empty(0), torch.empty(0)

        max_len = max([len(seq) for seq in feature_seqs])
        input_size = feature_seqs[0].size(-1)

        padded_seqs = [torch.cat([seq, torch.zeros(max_len - len(seq), input_size)], dim=0) for seq in feature_seqs]

        return torch.stack(padded_seqs), torch.stack(labels)

    """
    Wandelt eine einzelne Experience-Liste in einen flachen Feature-Vektor um
    """
    def months_between(self, start, end):
        delta = relativedelta(end, start)
        return delta.years * 12 + delta.months

    def extract_features_from_single_user(self, user_doc: dict):

        expected_features = 34  # z. B. 17 Positionen à 2 Werte
        experiences = user_doc.get("career_history", [])
        feature_seq = []

        for job in experiences:
            try:
                start = datetime.strptime(job["startDate"], "%Y-%m-%d")
                end = datetime.strptime(job["endDate"], "%Y-%m-%d") if job["endDate"].lower() != "present" else datetime.now()
                duration = self.months_between(start, end)
                level = self.get_position_level(job.get("position", ""))
                feature_seq.append([duration, level])
            except Exception as e:
                print(f"Fehler bei {job}: {e}")
                continue

        if not feature_seq:
            return np.zeros((1, expected_features), dtype=np.float32)

        flat = np.array(feature_seq, dtype=np.float32).flatten().reshape(1, -1)
        current_len = flat.shape[1]

        # Padding / Truncating
        if current_len < expected_features:
            pad = np.zeros((1, expected_features - current_len), dtype=np.float32)
            flat = np.hstack([flat, pad])
        elif current_len > expected_features:
            flat = flat[:, :expected_features]

        return flat