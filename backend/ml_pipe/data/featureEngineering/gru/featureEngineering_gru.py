import torch
import json
import os
from rapidfuzz import process, fuzz
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

'''
Feature Engineering for GRU
'''
class FeatureEngineering:

    '''
    Initialize FeatureEngineering
    '''
    def __init__(self):
        script_dir = os.path.dirname(__file__)

        position_level_path = os.path.join(script_dir, '..', 'position_level.json')
        with open(position_level_path, "r", encoding="utf-8") as f:
            self.position_levels = json.load(f)

        self.position_map = {
            entry["position"].lower(): (entry["level"], entry["branche"], entry["durchschnittszeit_tage"]) 
            for entry in self.position_levels
        }
        self.position_list = [entry["position"].lower() for entry in self.position_levels]

        self.branche_keywords = {
            "sales": ["sales", "vertrieb", "verkauf", "account", "business development"],
            "engineering": ["engineer", "developer", "software", "system", "tech", "it", "architect"],
            "consulting": ["consultant", "consulting", "berater", "beratung", "strategy"]
        }
        self.branche_set = set(entry["branche"].lower() for entry in self.position_levels)

        study_field_path = os.path.join(script_dir, '..', 'study_field_map.json')
        with open(study_field_path, "r", encoding="utf-8") as f:
            self.study_field_map = json.load(f)

        position_idx_path = os.path.join(script_dir, '..', '..', 'dataModule', 'tft', 'position_to_idx.json')
        with open(position_idx_path, "r", encoding="utf-8") as f:
            self.position_to_idx = json.load(f)

        self.company_size_map = {
            "small": 1,
            "medium": 2,
            "large": 3,
            "enterprise": 4
        }

    '''
    Get branche from keywords
    '''
    def get_branche_from_keywords(self, pos):
        pos_lower = pos.lower()
        for branche, keywords in self.branche_keywords.items():
            if any(keyword in pos_lower for keyword in keywords):
                return branche
        return None

    '''
    Map position to level, branche and average time
    '''
    def map_position(self, pos):
        if not pos:
            return 0, 0, 0
        pos_clean = pos.lower().strip()

        if pos_clean in self.position_map:
            level, branche, durchschnittszeit = self.position_map[pos_clean]
            return float(level), float(self.get_branche_num(branche)), float(durchschnittszeit)

        match, score, _ = process.extractOne(pos_clean, self.position_list, scorer=fuzz.ratio)

        if score >= 30:
            level, branche, durchschnittszeit = self.position_map[match]

            return float(level), float(self.get_branche_num(branche)), float(durchschnittszeit)

        return 0, 0, 0 # no match found

    '''
    Get branche number
    '''
    def get_branche_num(self, branche):
        branche_map = {
            "bau": 1,
            "consulting": 2,
            "customerservice": 3,
            "design": 4,
            "education": 5,
            "einkauf": 6,
            "engineering": 7,
            "finance": 8,
            "freelance": 9,
            "gesundheit": 10,
            "healthcare": 11,
            "hr": 12,
            "immobilien": 13,
            "it": 14,
            "legal": 15,
            "logistik": 16,
            "marketing": 17,
            "medien": 18,
            "operations": 19,
            "produktion": 20,
            "projektmanagement": 21,
            "research": 22,
            "sales": 23,
            "verwaltung": 24
        }
        return branche_map.get(branche.lower(), 0)

    '''
    Map branche from company_industry
    '''
    def map_branche(self, doc):
        company_industry = doc.get("company_industry", "")
        if company_industry:
            industry = company_industry.split(",")[0].strip().lower()
            if industry in self.branche_set:
                return self.get_branche_num(industry)
        pos = doc.get("aktuelle_position", "")
        return self.map_position(pos)[1]

    '''
    Get study field number
    '''
    def get_study_field_num(self, study_field):
        if not study_field:
            return 0
        field = study_field.lower().strip()
        for key in self.study_field_map:
            if key in field:
                return self.study_field_map[key]
        return 0

    '''
    Get position index
    '''
    def get_position_idx(self, pos):
        if not pos:
            return 0
        pos_clean = pos.lower().strip()
        return self.position_to_idx.get(pos_clean, 0)

    '''
    Get company size number
    '''
    def get_company_size_num(self, size):
        if not size:
            return 0
        return self.company_size_map.get(size.lower(), 0)

    '''
    Extract features and labels for training
    '''
    def extract_features_and_labels_for_training(self, documents):
        all_sequences = []
        all_labels = []
        for doc in documents:
            try:
                features = [
                    float(doc.get("berufserfahrung_bis_zeitpunkt", 0) or 0),
                    float(doc.get("anzahl_wechsel_bisher", 0) or 0),
                    float(doc.get("anzahl_jobs_bisher", 0) or 0),
                    float(doc.get("durchschnittsdauer_bisheriger_jobs", 0) or 0),
                    float(doc.get("highest_degree", 0) or 0),
                    float(doc.get("age_category", 0) or 0),
                    #float(doc.get("anzahl_standortwechsel", 0) or 0),
                    #float(self.get_study_field_num(doc.get("study_field", "")) or 0),
                    #float(self.get_company_size_num(doc.get("company_size_category", "")) or 0),
                ]

                level, branche, durchschnittszeit = self.map_position(doc.get("aktuelle_position", ""))
                features.extend([
                    float(level or 0),
                    float(branche or 0),
                    float(durchschnittszeit or 0)
                ])

                position_idx = self.get_position_idx(doc.get("aktuelle_position", ""))
                features.append(float(position_idx or 0))

                label = float(doc.get("label", 0) or 0)

                all_sequences.append([features])
                all_labels.append([label])

            except Exception as e:
                print(f"Error processing document: {str(e)}")
                print(f"Problem document: {doc}")
                continue

        if not all_sequences:
            raise ValueError("No valid sequences could be extracted")

        return (
            torch.tensor(all_sequences, dtype=torch.float32),
            torch.tensor(all_labels, dtype=torch.float32)
        )

    '''
    Extract sequences by profile
    '''
    def extract_sequences_by_profile(self, documents, min_seq_len=2):
        profile_groups = defaultdict(list)
        N = 2

        for doc in documents:
            profile_id = doc.get("profile_id")
            if profile_id:
                profile_groups[profile_id].append(doc)

        all_sequences = []
        all_labels = []

        for profile_id, entries in profile_groups.items():
            entries = sorted(entries, key=lambda x: x.get("zeitpunkt", 0))

            echte_positionen = []
            last_position = None
            for doc in entries:
                pos = doc.get("aktuelle_position", "")
                if pos != last_position:
                    echte_positionen.append({
                        "position": pos,
                        "branche": self.map_position(pos)[1],
                        "durchschnittsdauer": float(doc.get("durchschnittsdauer_bisheriger_jobs", 0) or 0),
                        "zeitpunkt": doc.get("zeitpunkt", 0)
                    })
                    last_position = pos

            for i, doc in enumerate(entries):
                try:
                    features = [
                        float(doc.get("berufserfahrung_bis_zeitpunkt", 0) or 0),
                        float(doc.get("anzahl_wechsel_bisher", 0) or 0),
                        float(doc.get("anzahl_jobs_bisher", 0) or 0),
                        float(doc.get("durchschnittsdauer_bisheriger_jobs", 0) or 0),
                        float(doc.get("highest_degree", 0) or 0),
                        float(doc.get("age_category", 0) or 0),
                    ]

                    level, branche, durchschnittszeit = self.map_position(doc.get("aktuelle_position", ""))
                    features.extend([level or 0, branche or 0, durchschnittszeit or 0])

                    position_idx = self.get_position_idx(doc.get("aktuelle_position", ""))
                    features.append(float(position_idx or 0))

                    current_time = doc.get("zeitpunkt", 0)
                    path_features = []
                    used_positions = set()
                    count = 0
                    for prev in reversed(echte_positionen):
                        if prev["zeitpunkt"] < current_time and prev["position"] not in used_positions:
                            path_features.extend([
                                float(self.map_position(prev["position"])[0]),  # Level
                                float(prev["branche"]),
                                float(prev["durchschnittsdauer"])
                            ])
                            used_positions.add(prev["position"])
                            count += 1
                            if count == N:
                                break

                    while count < N:
                        path_features.extend([0.0, 0.0, 0.0])
                        count += 1

                    features.extend(path_features)

                    label = float(doc.get("label", 0) or 0)

                    all_sequences.append([features])
                    all_labels.append([label])
                except Exception as e:
                    logging.warning(f"Error with {profile_id}: {e}")
                    continue

        if not all_sequences:
            raise ValueError("No valid sequences found.")

        return (
            torch.tensor(all_sequences, dtype=torch.float32),
            torch.tensor(all_labels, dtype=torch.float32)
        )

    '''
    Map position fuzzy
    '''
    def map_position_fuzzy(self, pos, threshold=80):
        if not pos:
            return 0, 0, 0

        pos_clean = pos.lower().strip()

        match, score, _ = process.extractOne(pos_clean, self.position_list, scorer=fuzz.ratio)

        if score >= threshold:
            level, branche, durchschnittszeit = self.position_map[match]

            return float(level), float(self.get_branche_num(branche)), float(durchschnittszeit)

        logger.warning(f"No match found for position: {pos}")
        return 0, 0, 0