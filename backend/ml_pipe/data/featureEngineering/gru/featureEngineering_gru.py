import torch
import json
import os

class FeatureEngineering:
    def __init__(self):
        json_path = os.path.join(
            "/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/data/featureEngineering/position_level.json"
        )
        with open(json_path, "r", encoding="utf-8") as f:
            self.position_levels = json.load(f)
        self.position_map = {entry["position"].lower(): (entry["level"], entry["branche"]) for entry in self.position_levels}

    def map_position(self, pos):
        if not pos:
            return 0, 0
        pos = pos.lower().strip()
        if pos in self.position_map:
            level, branche = self.position_map[pos]
        else:
            level, branche = 0, 0
        branche_map = {"sales": 1, "engineering": 2, "consulting": 3}
        branche_num = branche_map.get(branche, 0)
        return float(level), float(branche_num)

    def extract_features_and_labels_for_training(self, documents):
        all_sequences = []
        all_labels = []
        for doc in documents:
            features = [
                float(doc.get("berufserfahrung_bis_zeitpunkt", 0)),
                float(doc.get("anzahl_wechsel_bisher", 0)),
                float(doc.get("anzahl_jobs_bisher", 0)),
                float(doc.get("durchschnittsdauer_bisheriger_jobs", 0)),
                float(doc.get("zeitpunkt", 0)),
            ]
            level, branche = self.map_position(doc.get("aktuelle_position", ""))
            features.append(level)
            features.append(branche)
            all_sequences.append([features])
            all_labels.append([float(doc.get("label", 0))])
        return (
            torch.tensor(all_sequences, dtype=torch.float32),
            torch.tensor(all_labels, dtype=torch.float32)
        )