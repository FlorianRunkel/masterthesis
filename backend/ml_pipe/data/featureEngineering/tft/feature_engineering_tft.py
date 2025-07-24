import torch
import json
import os
import numpy as np
from rapidfuzz import process, fuzz
from collections import defaultdict
from datetime import datetime

'''
Feature Engineering for TFT
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
    Map position
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
            "bau": 1, "consulting": 2, "customerservice": 3, "design": 4, "education": 5, "einkauf": 6,
            "engineering": 7, "finance": 8, "freelance": 9, "gesundheit": 10, "healthcare": 11, "hr": 12,
            "immobilien": 13, "it": 14, "legal": 15, "logistik": 16, "marketing": 17, "medien": 18,
            "operations": 19, "produktion": 20, "projektmanagement": 21, "research": 22, "sales": 23, "verwaltung": 24
        }
        return float(branche_map.get(branche.lower(), 0))

    '''
    Get study field number
    '''
    def get_study_field_num(self, study_field):
        if not study_field:
            return 0.0
        field = study_field.lower().strip()
        for key in self.study_field_map:
            if key in field:
                return float(self.study_field_map[key])
        return 0.0 # no match found

    '''
    Get position index
    '''
    def get_position_idx(self, pos):
        if not pos:
            return 0
        pos_clean = pos.lower().strip()
        return int(self.position_to_idx.get(pos_clean, 0))

    '''
    Get company size number
    '''
    def get_company_size_num(self, size):
        if not size:
            return 0.0
        return float(self.company_size_map.get(size.lower(), 0))

    '''
    Group profiles by id
    '''
    def group_profiles_by_id(self, documents):
        profile_groups = defaultdict(list)
        for doc in documents:
            profile_id = doc.get("profile_id")
            if profile_id:
                profile_groups[profile_id].append(doc)
        return profile_groups

    '''
    Get max sequence length
    '''
    def get_max_sequence_length(self, profile_groups, min_seq_len):
        max_seq_len = 0
        for profile_id, entries in profile_groups.items():
            entries = sorted(entries, key=lambda x: x.get("zeitpunkt", 0))
            if len(entries) >= min_seq_len:
                max_seq_len = max(max_seq_len, len(entries))
        return max_seq_len

    '''
    Build position cache
    '''
    def build_position_cache(self, entries):
        position_cache = {}
        for doc in entries:
            pos = doc.get("aktuelle_position", "")
            if pos not in position_cache:
                level, branche, durchschnittszeit = self.map_position(pos)
                position_idx = self.get_position_idx(pos)
                position_cache[pos] = {
                    'level': float(level or 0),
                    'branche': float(branche or 0),
                    'durchschnittszeit': float(durchschnittszeit or 0),
                    'position_idx': float(position_idx)
                }
        return position_cache

    '''
    Extract true positions
    '''
    def extract_true_positions(self, entries, position_cache):
        echte_positionen = []
        last_position = None
        for doc in entries:
            pos = doc.get("aktuelle_position", "")
            if pos != last_position:
                echte_positionen.append({
                    "position": pos,
                    "branche": position_cache[pos]['branche'],
                    "durchschnittsdauer": float(doc.get("durchschnittsdauer_bisheriger_jobs", 0) or 0),
                    "zeitpunkt": doc.get("zeitpunkt", 0),
                    "level": position_cache[pos]['level']
                })
                last_position = pos
        return echte_positionen

    '''
    Extract features for document
    '''
    def extract_features_for_doc(self, doc, position_cache, echte_positionen):
        features = []
        # Basis-Features (6)
        features.extend([
            float(doc.get("berufserfahrung_bis_zeitpunkt", 0) or 0),
            float(doc.get("anzahl_wechsel_bisher", 0) or 0),
            float(doc.get("anzahl_jobs_bisher", 0) or 0),
            float(doc.get("durchschnittsdauer_bisheriger_jobs", 0) or 0),
            float(doc.get("highest_degree", 0) or 0),
            float(doc.get("age_category", 0) or 0),
        ])
        pos = doc.get("aktuelle_position", "")
        pos_data = position_cache.get(pos, {'level': 0.0, 'branche': 0.0, 'durchschnittszeit': 0.0, 'position_idx': 0.0})

        features.extend([
            pos_data['level'],
            pos_data['branche'],
            pos_data['durchschnittszeit']
        ])
        features.append(pos_data['position_idx'])
        zeitpunkt = doc.get("zeitpunkt", 0)
        if zeitpunkt > 0:
            dt = datetime.fromtimestamp(zeitpunkt)
            weekday = dt.weekday()
            month = dt.month
            features.extend([
                float(weekday),
                float(np.sin(2 * np.pi * weekday / 7)),
                float(np.cos(2 * np.pi * weekday / 7)),
                float(month),
                float(np.sin(2 * np.pi * month / 12)),
                float(np.cos(2 * np.pi * month / 12))
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        current_time = doc.get("zeitpunkt", 0)
        path_features = []
        used_positions = set()
        count = 0
        N = 2
        for prev in reversed(echte_positionen):
            if prev["zeitpunkt"] < current_time and prev["position"] not in used_positions:
                path_features.extend([
                    prev["level"],
                    prev["branche"],
                    prev["durchschnittsdauer"]
                ])
                used_positions.add(prev["position"])
                count += 1
                if count == N:
                    break
        while count < N:
            path_features.extend([0.0, 0.0, 0.0])
            count += 1
        features.extend(path_features)

        company_size = doc.get("company_size_category", "medium")
        company_size_num = self.get_company_size_num(company_size)
        features.append(company_size_num)

        study_field = doc.get("study_field", "Informatics")
        study_field_num = self.get_study_field_num(study_field)
        features.append(study_field_num)
        return features

    '''
    Pad sequences
    '''
    def pad_sequences(self, sequences, target_length, feature_dim=24):
        padded_sequences = []
        for sequence in sequences:
            padded_seq = sequence.copy()
            while len(padded_seq) < target_length:
                padding_features = [0.0] * feature_dim
                padded_seq.append(padding_features)
            padded_sequences.append(padded_seq)
        return padded_sequences

    '''
    Build sequences and labels
    '''
    def build_sequences_and_labels(self, profile_groups, min_seq_len):
        all_sequences = []
        all_labels = []
        all_positions = []
        processed_profiles = 0
        for profile_id, entries in profile_groups.items():
            entries = sorted(entries, key=lambda x: x.get("zeitpunkt", 0))
            if len(entries) < min_seq_len:
                continue
            position_cache = self.build_position_cache(entries)
            echte_positionen = self.extract_true_positions(entries, position_cache)
            sequence = []
            positions = []
            for doc in entries:
                try:
                    features = self.extract_features_for_doc(doc, position_cache, echte_positionen)
                    sequence.append(features)
                    positions.append(doc.get("aktuelle_position", "unknown"))
                except Exception as e:
                    print(f"[WARN] Error with {profile_id}: {e}")
                    continue
            if sequence:
                filtered_sequence = []
                filtered_positions = []
                for features, position in zip(sequence, positions):
                    non_zero_features = [f for f in features if f != 0.0]
                    if len(non_zero_features) > 0:
                        filtered_sequence.append(features)
                        filtered_positions.append(position)
                if filtered_sequence:
                    all_sequences.append(filtered_sequence)
                    all_positions.append(filtered_positions)
                    all_labels.append(float(entries[-1].get("label", 0) or 0))
                processed_profiles += 1
        return all_sequences, all_labels, all_positions

    '''
    Extract sequences by profile
    '''
    def extract_sequences_by_profile(self, documents, min_seq_len=2):
        profile_groups = self.group_profiles_by_id(documents)
        max_seq_len = self.get_max_sequence_length(profile_groups, min_seq_len)
        all_sequences, all_labels, all_positions = self.build_sequences_and_labels(profile_groups, min_seq_len)
        if not all_sequences:
            raise ValueError("No valid sequences found.")
        padded_sequences = self.pad_sequences(all_sequences, max_seq_len, feature_dim=24)
        return (
            torch.tensor(padded_sequences, dtype=torch.float32),
            torch.tensor(all_labels, dtype=torch.float32),
            all_positions
        )