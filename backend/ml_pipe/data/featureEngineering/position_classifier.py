from typing import Tuple, List, Dict, Any
import os
import json
from difflib import SequenceMatcher

class PositionClassifier:
    def __init__(self):
        """Initialisiert den PositionClassifier mit der position_level.json"""
        # Lade die Position-Level-Zuordnungen
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "position_level.json")
        
        with open(json_path, "r", encoding="utf-8") as f:
            self.position_data = json.load(f)
            
        # Erstelle Index für schnelleres Matching
        self.position_index = {}
        for entry in self.position_data:
            position = entry["position"].lower()
            self.position_index[position] = {
                "level": entry["level"],
                "branche": entry["branche"]
            }
            
        # Keywords für zusätzliches Matching
        self.level_keywords = {
            1: ["junior", "trainee", "intern", "praktikant", "werkstudent", "student", "assistant"],
            2: ["junior", "associate", "entry"],
            3: ["", "regular", "intermediate"],  # Leerer String für Standard-Level
            4: ["senior", "expert", "specialist", "lead", "experienced"],
            5: ["team lead", "architect", "principal", "group lead"],
            6: ["manager", "head", "director"],
            7: ["head of", "director", "chief"],
            8: ["vp", "vice president", "chief", "cto", "ceo"]
        }
        
        self.branch_keywords = {
            1: ["sales", "account", "business development", "vertrieb", "verkauf", "customer", "revenue"],
            2: ["engineer", "developer", "software", "frontend", "backend", "fullstack", "devops", "tech", "qa", "test"],
            3: ["consult", "consulting", "strategy", "project", "advisor", "analyst"]
        }
        
    def get_similarity(self, a: str, b: str) -> float:
        """Berechnet die Ähnlichkeit zwischen zwei Strings"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
        
    def find_best_match(self, title: str) -> Tuple[int, int]:
        """
        Findet die beste Übereinstimmung für einen Positionstitel.
        
        Args:
            title: Der zu klassifizierende Positionstitel
            
        Returns:
            Tuple[int, int]: (level, branche)
        """
        if not title:
            return 2, 0  # Default: Regular Level, Other
            
        title = title.lower().strip()
        
        # 1. Exakte Übereinstimmung
        if title in self.position_index:
            match = self.position_index[title]
            return match["level"], match["branche"]
            
        # 2. Teilstring-Matching
        for pos in self.position_index:
            if pos in title or title in pos:
                match = self.position_index[pos]
                return match["level"], match["branche"]
                
        # 3. Keyword-basierte Analyse
        level = self._determine_level(title)
        branche = self._determine_branche(title)
        
        return level, branche
        
    def _determine_level(self, title: str) -> int:
        """Bestimmt das Level basierend auf Keywords"""
        max_level = 2  # Default Level
        
        # Prüfe Level-Keywords
        for level, keywords in self.level_keywords.items():
            for keyword in keywords:
                if keyword and keyword in title:
                    max_level = max(max_level, level)
                    
        return max_level
        
    def _determine_branche(self, title: str) -> int:
        """Bestimmt die Branche basierend auf Keywords"""
        max_score = 0
        best_branche = 0  # Default: Other
        
        # Prüfe Branchen-Keywords
        for branche, keywords in self.branch_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in title:
                    score += 1
            if score > max_score:
                max_score = score
                best_branche = branche
                
        return best_branche
        
    def classify_position(self, title: str, company: str = None, description: str = None) -> Tuple[int, int]:
        """
        Klassifiziert eine Position basierend auf Titel und optionalem Kontext.
        
        Args:
            title: Der Positionstitel
            company: (Optional) Der Firmenname für zusätzlichen Kontext
            description: (Optional) Die Positionsbeschreibung für zusätzlichen Kontext
            
        Returns:
            Tuple[int, int]: (level, branche)
        """
        try:
            # Hauptklassifizierung basierend auf Titel
            level, branche = self.find_best_match(title)
            
            # Wenn Beschreibung vorhanden, nutze sie für bessere Branchenerkennung
            if description and branche == 0:
                _, branche_from_desc = self.find_best_match(description)
                if branche_from_desc != 0:
                    branche = branche_from_desc
                    
            return level, branche
            
        except Exception as e:
            print(f"Fehler bei der Klassifizierung von '{title}': {str(e)}")
            return 2, 0  # Default-Werte im Fehlerfall
            
    def get_level_description(self, level: int) -> str:
        """Gibt die Beschreibung für ein Level zurück."""
        levels = {
            1: "Junior/Entry Level",
            2: "Regular/Mid Level",
            3: "Senior Level",
            4: "Expert Level",
            5: "Team Lead Level",
            6: "Management Level",
            7: "Director Level",
            8: "Executive Level"
        }
        return levels.get(level, "Unbekanntes Level")
        
    def get_branch_description(self, branch: int) -> str:
        """Gibt die Beschreibung für eine Branche zurück."""
        branches = {
            0: "Other",
            1: "Sales",
            2: "Engineering/IT",
            3: "Consulting"
        }
        return branches.get(branch, "Unbekannte Branche") 