import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Dict

class LinkedInDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def process_raw_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Verarbeitet die rohen LinkedIn-Daten"""
        # Hier spezifische Verarbeitungsschritte für LinkedIn-Daten
        # Beispiel:
        # - Entfernen von Duplikaten
        # - Behandlung fehlender Werte
        # - Konvertierung von Datumsangaben
        # - Extraktion von Features aus Textfeldern
        
        processed_data = data.copy()
        
        # Beispiel für Datumsverarbeitung
        if 'created_at' in processed_data.columns:
            processed_data['created_at'] = pd.to_datetime(processed_data['created_at'])
            processed_data['account_age_days'] = (pd.Timestamp.now() - processed_data['created_at']).dt.days
        
        # Beispiel für Textfeld-Verarbeitung
        if 'headline' in processed_data.columns:
            processed_data['headline_length'] = processed_data['headline'].str.len()
        
        return processed_data
    
    def prepare_features(self, data: pd.DataFrame, categorical_columns: List[str], 
                        numerical_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Bereitet Features für das ML-Modell vor"""
        # Verarbeitung kategorischer Variablen
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            data[col] = self.label_encoders[col].fit_transform(data[col])
        
        # Verarbeitung numerischer Variablen
        numerical_data = data[numerical_columns].copy()
        numerical_data = self.scaler.fit_transform(numerical_data)
        
        return numerical_data, data[categorical_columns].values
    
    def create_feature_matrix(self, numerical_features: np.ndarray, 
                            categorical_features: np.ndarray) -> np.ndarray:
        """Erstellt die finale Feature-Matrix"""
        return np.hstack([numerical_features, categorical_features])
    
    def inverse_transform_predictions(self, predictions: np.ndarray, 
                                   target_column: str) -> np.ndarray:
        """Transformiert Vorhersagen zurück in das ursprüngliche Format"""
        if target_column in self.label_encoders:
            return self.label_encoders[target_column].inverse_transform(predictions)
        return predictions 