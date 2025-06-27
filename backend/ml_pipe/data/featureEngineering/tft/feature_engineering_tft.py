import torch
import json
import os
import numpy as np
from rapidfuzz import process, fuzz
from collections import defaultdict
from datetime import datetime

class FeatureEngineering:
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
        return 0, 0, 0

    def get_branche_num(self, branche):
        branche_map = {
            "bau": 1, "consulting": 2, "customerservice": 3, "design": 4, "education": 5, "einkauf": 6,
            "engineering": 7, "finance": 8, "freelance": 9, "gesundheit": 10, "healthcare": 11, "hr": 12,
            "immobilien": 13, "it": 14, "legal": 15, "logistik": 16, "marketing": 17, "medien": 18,
            "operations": 19, "produktion": 20, "projektmanagement": 21, "research": 22, "sales": 23, "verwaltung": 24
        }
        return float(branche_map.get(branche.lower(), 0))

    def get_study_field_num(self, study_field):
        if not study_field:
            return 0.0
        field = study_field.lower().strip()
        for key in self.study_field_map:
            if key in field:
                return float(self.study_field_map[key])
        return 0.0

    def get_position_idx(self, pos):
        if not pos:
            return 0
        pos_clean = pos.lower().strip()
        return int(self.position_to_idx.get(pos_clean, 0))

    def get_company_size_num(self, size):
        if not size:
            return 0.0
        return float(self.company_size_map.get(size.lower(), 0))

    def extract_sequences_by_profile(self, documents, min_seq_len=2):
        profile_groups = defaultdict(list)
        for doc in documents:
            profile_id = doc.get("profile_id")
            if profile_id:
                profile_groups[profile_id].append(doc)
        all_sequences = []
        all_labels = []
        all_positions = []
        
        # Finde die maximale Sequenzlänge
        max_seq_len = 0
        for profile_id, entries in profile_groups.items():
            entries = sorted(entries, key=lambda x: x.get("zeitpunkt", 0))
            if len(entries) >= min_seq_len:
                max_seq_len = max(max_seq_len, len(entries))
        
        print(f"Maximale Sequenzlänge: {max_seq_len}")
        print(f"Anzahl Profile: {len(profile_groups)}")
        
        # Debug: Zeige erste paar Dokumente
        print("\nDEBUG - Erste Dokumente:")
        for i, doc in enumerate(documents[:3]):
            print(f"Dokument {i}:")
            print(f"  profile_id: {doc.get('profile_id')}")
            print(f"  aktuelle_position: {doc.get('aktuelle_position')}")
            print(f"  berufserfahrung_bis_zeitpunkt: {doc.get('berufserfahrung_bis_zeitpunkt')}")
            print(f"  anzahl_wechsel_bisher: {doc.get('anzahl_wechsel_bisher')}")
            print(f"  zeitpunkt: {doc.get('zeitpunkt')}")
            print(f"  label: {doc.get('label')}")
        
        processed_profiles = 0
        for profile_id, entries in profile_groups.items():
            entries = sorted(entries, key=lambda x: x.get("zeitpunkt", 0))
            if len(entries) < min_seq_len:
                continue
            
            # Debug: Zeige erste paar Profile
            if processed_profiles < 3:
                print(f"\nDEBUG - Profile {profile_id}:")
                print(f"  Anzahl Einträge: {len(entries)}")
                print(f"  Erste Einträge:")
                for i, entry in enumerate(entries[:3]):
                    print(f"    Eintrag {i}: position={entry.get('aktuelle_position')}, zeitpunkt={entry.get('zeitpunkt')}")
            
            # 1. Cache für Position-Mappings (nur einmal berechnen)
            position_cache = {}
            echte_positionen = []
            last_position = None
            
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
                
                if pos != last_position:
                    echte_positionen.append({
                        "position": pos,
                        "branche": position_cache[pos]['branche'],
                        "durchschnittsdauer": float(doc.get("durchschnittsdauer_bisheriger_jobs", 0) or 0),
                        "zeitpunkt": doc.get("zeitpunkt", 0),
                        "level": position_cache[pos]['level']
                    })
                    last_position = pos
            
            # Debug: Zeige echte Positionen
            if processed_profiles < 3:
                print(f"  Echte Positionen: {len(echte_positionen)}")
                for pos in echte_positionen[:3]:
                    print(f"    {pos}")
            
            sequence = []
            positions = []
            
            for doc in entries:
                try:
                    # Debug: Zeige Dokument-Daten
                    if processed_profiles < 3 and len(sequence) < 3:
                        print(f"\n  DEBUG - Doc {len(sequence)}:")
                        print(f"    position: {doc.get('aktuelle_position')}")
                        print(f"    berufserfahrung: {doc.get('berufserfahrung_bis_zeitpunkt')}")
                        print(f"    anzahl_wechsel: {doc.get('anzahl_wechsel_bisher')}")
                        print(f"    zeitpunkt: {doc.get('zeitpunkt')}")
                    
                    # Hole Position-Daten aus Cache
                    pos = doc.get("aktuelle_position", "")
                    pos_data = position_cache.get(pos, {
                        'level': 0.0, 'branche': 0.0, 'durchschnittszeit': 0.0, 'position_idx': 0.0
                    })
                    
                    # Basis-Features (6)
                    features = [
                        float(doc.get("berufserfahrung_bis_zeitpunkt", 0) or 0),
                        float(doc.get("anzahl_wechsel_bisher", 0) or 0),
                        float(doc.get("anzahl_jobs_bisher", 0) or 0),
                        float(doc.get("durchschnittsdauer_bisheriger_jobs", 0) or 0),
                        float(doc.get("highest_degree", 0) or 0),
                        float(doc.get("age_category", 0) or 0),
                    ]
                    
                    if processed_profiles < 3 and len(sequence) < 3:
                        print(f"    Basis-Features: {features}")
                    
                    # Position-Features (3) - aus Cache
                    features.extend([
                        pos_data['level'],
                        pos_data['branche'],
                        pos_data['durchschnittszeit']
                    ])
                    
                    if processed_profiles < 3 and len(sequence) < 3:
                        print(f"    Position-Features: level={pos_data['level']}, branche={pos_data['branche']}, durchschnittszeit={pos_data['durchschnittszeit']}")
                    
                    # Position-ID (1) - aus Cache
                    features.append(pos_data['position_idx'])
                    
                    if processed_profiles < 3 and len(sequence) < 3:
                        print(f"    Position-ID: {pos_data['position_idx']}")
                        print(f"    Position: '{pos}'")
                        print(f"    Position-Cache: {position_cache.get(pos, 'Nicht gefunden')}")
                    
                    # Debug: Zeige Position-Mapping für erste paar Profile
                    if processed_profiles < 3 and len(sequence) < 3:
                        print(f"    DEBUG - Position-Mapping:")
                        print(f"      Position: '{pos}'")
                        print(f"      get_position_idx Ergebnis: {self.get_position_idx(pos)}")
                        print(f"      position_to_idx Keys (erste 5): {list(self.position_to_idx.keys())[:5]}")
                        print(f"      Position in position_to_idx: {pos.lower().strip() in self.position_to_idx}")
                    
                    # Zeit-Features (6)
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
                    
                    if processed_profiles < 3 and len(sequence) < 3:
                        print(f"    Zeit-Features: zeitpunkt={zeitpunkt}, weekday={weekday if zeitpunkt > 0 else 'N/A'}")
                    
                    # Karrierepfad-Features (6) - nur die letzten 2 Positionen
                    current_time = doc.get("zeitpunkt", 0)
                    path_features = []
                    used_positions = set()
                    count = 0
                    N = 2  # Nur die letzten 2 Positionen
                    
                    for prev in reversed(echte_positionen):
                        if prev["zeitpunkt"] < current_time and prev["position"] not in used_positions:
                            path_features.extend([
                                prev["level"],  # Aus Cache
                                prev["branche"],
                                prev["durchschnittsdauer"]
                            ])
                            used_positions.add(prev["position"])
                            count += 1
                            if count == N:
                                break
                    
                    # Padding falls weniger als N gefunden
                    while count < N:
                        path_features.extend([0.0, 0.0, 0.0])
                        count += 1
                    
                    features.extend(path_features)
                    
                    if processed_profiles < 3 and len(sequence) < 3:
                        print(f"    Karrierepfad-Features (erste 6): {path_features[:6]}")
                        print(f"    Finale Features (erste 10): {features[:10]}")
                        print(f"    Feature-Summe: {sum(features)}")
                        print(f"    Feature-Anzahl: {len(features)}")
                    
                    sequence.append(features)
                    positions.append(pos if pos else "unknown")
                    
                except Exception as e:
                    print(f"[WARN] Fehler bei {profile_id}: {e}")
                    continue
            
            # Filtere Padding-Zeilen (alle Features = 0)
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
        
        if not all_sequences:
            raise ValueError("Keine gültigen Sequenzen gefunden.")
        
        print(f"\nAnzahl Sequenzen: {len(all_sequences)}")
        print(f"Sequenzlänge: {len(all_sequences[0])}")
        print(f"Feature-Anzahl: {len(all_sequences[0][0])}")
        
        # Finde die maximale Sequenzlänge für Padding
        max_seq_len = max(len(seq) for seq in all_sequences)
        print(f"Maximale Sequenzlänge nach Filterung: {max_seq_len}")
        
        # Padding für Tensor-Konvertierung
        padded_sequences = []
        for sequence in all_sequences:
            padded_seq = sequence.copy()
            while len(padded_seq) < max_seq_len:
                padding_features = [0.0] * 22  # 22 Features
                padded_seq.append(padding_features)
            padded_sequences.append(padded_seq)
        
        # Debug: Zeige Feature-Statistiken
        print("\nDEBUG - Feature-Statistiken:")
        for i in range(min(5, len(padded_sequences[0][0]))):
            feature_values = [seq[0][i] for seq in padded_sequences if len(seq) > 0]
            print(f"Feature {i}: min={min(feature_values)}, max={max(feature_values)}, mean={sum(feature_values)/len(feature_values):.2f}")
        
        return (
            torch.tensor(padded_sequences, dtype=torch.float32),
            torch.tensor(all_labels, dtype=torch.float32),
            all_positions
        ) 