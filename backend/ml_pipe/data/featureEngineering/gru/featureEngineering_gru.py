import torch
import json
import os
from rapidfuzz import process, fuzz

class FeatureEngineering:
    def __init__(self):
        json_path = os.path.join(
            "/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/data/featureEngineering/position_level.json"
        )
        with open(json_path, "r", encoding="utf-8") as f:
            self.position_levels = json.load(f)
        self.position_map = {
            entry["position"].lower(): (entry["level"], entry["branche"], entry["durchschnittszeit_tage"]) 
            for entry in self.position_levels
        }
        self.position_list = [entry["position"].lower() for entry in self.position_levels]
        
        # Erstelle eine Liste von Schlüsselwörtern für jede Branche
        self.branche_keywords = {
            "sales": ["sales", "vertrieb", "verkauf", "account", "business development"],
            "engineering": ["engineer", "developer", "software", "system", "tech", "it", "architect"],
            "consulting": ["consultant", "consulting", "berater", "beratung", "strategy"]
        }

    def get_branche_from_keywords(self, pos):
        """Ermittelt die Branche basierend auf Schlüsselwörtern."""
        pos_lower = pos.lower()
        for branche, keywords in self.branche_keywords.items():
            if any(keyword in pos_lower for keyword in keywords):
                return branche
        return None

    def map_position(self, pos):
        if not pos:
            return 0, 0, 0
        
        pos_clean = pos.lower().strip()
        
        # Direktes Mapping versuchen
        if pos_clean in self.position_map:
            level, branche, durchschnittszeit = self.position_map[pos_clean]
            print(f"Exaktes Match gefunden für Position: '{pos}'")
            return float(level), float(self.get_branche_num(branche)), float(durchschnittszeit)
        
        # Fuzzy Matching mit Schwellenwert von 30
        match, score, _ = process.extractOne(pos_clean, self.position_list, scorer=fuzz.ratio)
        
        if score >= 30:
            level, branche, durchschnittszeit = self.position_map[match]
            print(f"Fuzzy Match gefunden: '{pos}' -> '{match}' (Score: {score})")
            return float(level), float(self.get_branche_num(branche)), float(durchschnittszeit)
        
        print(f"Kein Match gefunden für Position: '{pos}'")
        return 0, 0, 0

    def get_branche_num(self, branche):
        """Konvertiert Branchennamen in numerische Werte."""
        branche_map = {"sales": 1, "engineering": 2, "consulting": 3}
        return branche_map.get(branche, 0)

    def extract_features_and_labels_for_training(self, documents):
        all_sequences = []
        all_labels = []
        for doc in documents:
            features = [
                float(doc.get("berufserfahrung_bis_zeitpunkt", 0)),
                float(doc.get("anzahl_wechsel_bisher", 0)),
                float(doc.get("anzahl_jobs_bisher", 0)),
                float(doc.get("durchschnittsdauer_bisheriger_jobs", 0)),
            ]
            level, branche, durchschnittszeit = self.map_position(doc.get("aktuelle_position", ""))
            features.append(level)
            features.append(branche)
            features.append(durchschnittszeit)
            all_sequences.append([features])
            all_labels.append([float(doc.get("label", 0))])
        return (
            torch.tensor(all_sequences, dtype=torch.float32),
            torch.tensor(all_labels, dtype=torch.float32)
        )