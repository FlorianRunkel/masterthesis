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

        # Fallback für unbekannte Positionen - basierend auf Schlüsselwörtern
        if any(keyword in pos_clean for keyword in ['thesis', 'master', 'bachelor', 'phd', 'research', 'student']):
            return 1, 'research', 180  # Student/Research Position
        elif any(keyword in pos_clean for keyword in ['intern', 'trainee', 'assistant']):
            return 1, 'engineering', 180  # Entry Level
        else:
            return 1, 'engineering', 300  # Default Entry Level

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