import torch
import json
import os
from rapidfuzz import process, fuzz
from collections import defaultdict

class FeatureEngineering:
    def __init__(self):
        # Lade Konfigurationsdateien relativ zum Skriptpfad
        script_dir = os.path.dirname(__file__)
        json_path = os.path.join(
            script_dir, '..', '..', 'data', 'featureEngineering', 'position_level.json'
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
        self.branche_set = set(entry["branche"].lower() for entry in self.position_levels)

        study_field_path = os.path.join(
            script_dir, '..', '..', 'data', 'featureEngineering', 'study_field_map.json'
        )
        with open(study_field_path, "r", encoding="utf-8") as f:
            self.study_field_map = json.load(f)

        # Position-zu-Index Mapping laden
        position_idx_path = os.path.join(
            script_dir, '..', '..', 'data', 'dataModule', 'tft', 'position_to_idx.json'
        )
        with open(position_idx_path, "r", encoding="utf-8") as f:
            self.position_to_idx = json.load(f)

        # Mapping für Unternehmensgrößen
        self.company_size_map = {
            "small": 1,
            "medium": 2,
            "large": 3,
            "enterprise": 4
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
            #print(f"Exaktes Match gefunden für Position: '{pos}'")
            return float(level), float(self.get_branche_num(branche)), float(durchschnittszeit)
        
        # Fuzzy Matching mit Schwellenwert von 30
        match, score, _ = process.extractOne(pos_clean, self.position_list, scorer=fuzz.ratio)
        
        if score >= 30:
            level, branche, durchschnittszeit = self.position_map[match]
            
            #print(f"Fuzzy Match gefunden: '{pos}' -> '{match}' (Score: {score})")
            return float(level), float(self.get_branche_num(branche)), float(durchschnittszeit)
        
        #print(f"Kein Match gefunden für Position: '{pos}'")
        return 0, 0, 0

    def get_branche_num(self, branche):
        """Konvertiert Branchennamen in numerische Werte."""
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

    def map_branche(self, doc):
        # 1. Company-Industry bevorzugen, falls vorhanden und bekannt
        company_industry = doc.get("company_industry", "")
        if company_industry:
            # Falls mehrere Branchen als String: nur die erste nehmen
            industry = company_industry.split(",")[0].strip().lower()
            if industry in self.branche_set:
                return self.get_branche_num(industry)
        # 2. Fallback: wie bisher über Position
        pos = doc.get("aktuelle_position", "")
        return self.map_position(pos)[1]  # gibt die Branchen-Nummer zurück

    def get_study_field_num(self, study_field):
        if not study_field:
            return 0
        field = study_field.lower().strip()
        for key in self.study_field_map:
            if key in field:
                return self.study_field_map[key]
        return 0

    def get_position_idx(self, pos):
        """Gibt die ID der Position zurück (oder 0, falls nicht gefunden)."""
        if not pos:
            return 0
        pos_clean = pos.lower().strip()
        return self.position_to_idx.get(pos_clean, 0)

    def get_company_size_num(self, size):
        """Konvertiert Unternehmensgrößen-Strings in numerische Werte."""
        if not size:
            return 0
        return self.company_size_map.get(size.lower(), 0)

    def extract_features_and_labels_for_training(self, documents):
        all_sequences = []
        all_labels = []
        for doc in documents:
            try:
                # Basis-Features mit Fehlerbehandlung
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

                # Position-bezogene Features mit Fehlerbehandlung
                level, branche, durchschnittszeit = self.map_position(doc.get("aktuelle_position", ""))
                features.extend([
                    float(level or 0),
                    float(branche or 0),
                    float(durchschnittszeit or 0)
                ])

                # Positions-ID mit Fehlerbehandlung
                position_idx = self.get_position_idx(doc.get("aktuelle_position", ""))
                features.append(float(position_idx or 0))

                # Label mit Fehlerbehandlung
                label = float(doc.get("label", 0) or 0)

                all_sequences.append([features])
                all_labels.append([label])

            except Exception as e:
                print(f"Fehler bei der Verarbeitung eines Dokuments: {str(e)}")
                print(f"Problemdokument: {doc}")
                continue

        if not all_sequences:
            raise ValueError("Keine gültigen Sequenzen konnten extrahiert werden")

        return (
            torch.tensor(all_sequences, dtype=torch.float32),
            torch.tensor(all_labels, dtype=torch.float32)
        )
    
    def extract_sequences_by_profile(self, documents, min_seq_len=2):
        profile_groups = defaultdict(list)

        for doc in documents:
            profile_id = doc.get("profile_id")
            if profile_id:
                profile_groups[profile_id].append(doc)

        all_sequences = []
        all_labels = []

        for profile_id, entries in profile_groups.items():
            # Sortiere die Zeitpunkte chronologisch
            entries = sorted(entries, key=lambda x: x.get("zeitpunkt", 0))

            # Optional: überspringe zu kurze Verläufe
            if len(entries) < min_seq_len:
                continue

            sequence = []
            for doc in entries:
                try:
                    # Basis-Features
                    features = [
                        float(doc.get("berufserfahrung_bis_zeitpunkt", 0) or 0),
                        float(doc.get("anzahl_wechsel_bisher", 0) or 0),
                        float(doc.get("anzahl_jobs_bisher", 0) or 0),
                        float(doc.get("durchschnittsdauer_bisheriger_jobs", 0) or 0),
                        float(doc.get("highest_degree", 0) or 0),
                        float(doc.get("age_category", 0) or 0),
                    ]

                    # Positionen + Mapping
                    level, branche, durchschnittszeit = self.map_position(doc.get("aktuelle_position", ""))
                    features.extend([level or 0, branche or 0, durchschnittszeit or 0])

                    position_idx = self.get_position_idx(doc.get("aktuelle_position", ""))
                    features.append(float(position_idx or 0))

                    sequence.append(features)
                except Exception as e:
                    print(f"[WARN] Fehler bei {profile_id}: {e}")
                    continue

            if sequence:
                all_sequences.append(sequence)
                all_labels.append(float(entries[-1].get("label", 0) or 0))  # Label vom letzten Zeitpunkt

        if not all_sequences:
            raise ValueError("Keine gültigen Sequenzen gefunden.")

        return (
            torch.tensor(all_sequences, dtype=torch.float32),
            torch.tensor(all_labels, dtype=torch.float32)
        )

    def extract_sequences_by_profile_new(self, documents, min_seq_len=2):
        profile_groups = defaultdict(list)
        N = 2

        for doc in documents:
            profile_id = doc.get("profile_id")
            if profile_id:
                profile_groups[profile_id].append(doc)

        all_sequences = []
        all_labels = []

        for profile_id, entries in profile_groups.items():
            # Sortiere die Zeitpunkte chronologisch
            entries = sorted(entries, key=lambda x: x.get("zeitpunkt", 0))

            # 1. Liste der echten Positionswechsel (ohne Duplikate, chronologisch)
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
                    # Basis-Features
                    features = [
                        float(doc.get("berufserfahrung_bis_zeitpunkt", 0) or 0),
                        float(doc.get("anzahl_wechsel_bisher", 0) or 0),
                        float(doc.get("anzahl_jobs_bisher", 0) or 0),
                        float(doc.get("durchschnittsdauer_bisheriger_jobs", 0) or 0),
                        float(doc.get("highest_degree", 0) or 0),
                        float(doc.get("age_category", 0) or 0),
                    ]

                    # Positionen + Mapping
                    level, branche, durchschnittszeit = self.map_position(doc.get("aktuelle_position", ""))
                    features.extend([level or 0, branche or 0, durchschnittszeit or 0])

                    position_idx = self.get_position_idx(doc.get("aktuelle_position", ""))
                    features.append(float(position_idx or 0))

                    # 2. Karrierepfad-Feature: letzte N echte Positionen vor aktuellem Zeitpunkt
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
                    # Padding falls weniger als N gefunden
                    while count < N:
                        path_features.extend([0.0, 0.0, 0.0])
                        count += 1

                    features.extend(path_features)

                    # Label mit Fehlerbehandlung
                    label = float(doc.get("label", 0) or 0)

                    all_sequences.append([features])
                    all_labels.append([label])
                except Exception as e:
                    print(f"[WARN] Fehler bei {profile_id}: {e}")
                    continue

        if not all_sequences:
            raise ValueError("Keine gültigen Sequenzen gefunden.")

        return (
            torch.tensor(all_sequences, dtype=torch.float32),
            torch.tensor(all_labels, dtype=torch.float32)
        )

    def map_position_fuzzy(self, pos, threshold=80):
        if not pos:
            return 0, 0, 0
        
        pos_clean = pos.lower().strip()
        
        # Fuzzy Matching mit Schwellenwert von 80
        match, score, _ = process.extractOne(pos_clean, self.position_list, scorer=fuzz.ratio)
        
        if score >= threshold:
            level, branche, durchschnittszeit = self.position_map[match]
            
            #print(f"Fuzzy Match gefunden: '{pos}' -> '{match}' (Score: {score})")
            return float(level), float(self.get_branche_num(branche)), float(durchschnittszeit)
        
        #print(f"Kein Match gefunden für Position: '{pos}'")
        return 0, 0, 0