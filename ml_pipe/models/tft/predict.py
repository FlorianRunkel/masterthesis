import torch
import os
import shap
import numpy as np
from ml_pipe.data.featureEngineering.featureEngineering import featureEngineering
from ml_pipe.models.tft.model import TFTModel
from datetime import datetime
import traceback
import logging
from functools import partial

def preprocess(user_data):
    fe = featureEngineering()
    features = fe.extract_features_from_single_user(user_data)
    
    if features is None:
        raise ValueError("Nicht genug Daten für Vorhersage")

    # Die Features sind bereits in der richtigen Form [normalized_duration, level, branche_code]
    # Keine weitere Selektion nötig, da wir nur diese 3 Features verwenden
    return features.tolist()

def wrap_model_prediction(model, x):
    """Wrapper-Funktion für die Modellvorhersage, die für SHAP benötigt wird"""
    with torch.no_grad():
        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x).float()
        else:
            x_tensor = torch.tensor(x).float()
        return model(x_tensor).detach().numpy().reshape(-1, 1)

def predict(input_data, model_path="ml_pipe/models/tft/saved_models/tft_model_20250415_155249.pt"):
    try:
        # Initialisiere Feature Engineering
        fe = featureEngineering()
        
        # Extrahiere Features
        features = fe.extract_features_from_single_user(input_data)
        if features is None:
            return {
                "confidence": [0.0], 
                "recommendations": ["Keine gültigen Karrieredaten vorhanden"],
                "explanations": []
            }
            
        # Modell initialisieren
        model = TFTModel(input_size=3, hidden_size=32)
        
        # Modell laden
        checkpoint = torch.load(model_path, weights_only=True)
        model.load_state_dict(checkpoint)
        model.eval()
        
        # Input vorbereiten
        input_tensor = torch.from_numpy(features).float()
        
        # Vorhersage machen
        with torch.no_grad():
            pred = model(input_tensor)
        
        # Vorhersage interpretieren
        pred_value = float(pred.item())
        
        # SHAP-Werte berechnen
        background = np.zeros((1, 3))  # Reduzierter Hintergrunddatensatz
        # Verwende die neue Wrapper-Funktion mit partial
        model_wrapper = partial(wrap_model_prediction, model)
        explainer = shap.KernelExplainer(model_wrapper, background)
        shap_values = explainer.shap_values(features.reshape(1, -1))[0]
        
        # Feature-Namen für die Erklärungen
        feature_names = ['Beschäftigungsdauer', 'Positionslevel', 'Branchenkontinuität']
        
        # Erstelle Erklärungen und Features Dictionary
        explanations = []
        
        # Berechne die Summe der absoluten SHAP-Werte für die Prozentberechnung
        total_impact = sum(abs(float(value)) for value in shap_values)
        
        for idx, (value, feature_name, feature_val) in enumerate(zip(shap_values, feature_names, features)):
            impact = float(value)
            # Berechne den prozentualen Einfluss
            impact_percentage = (abs(impact) / total_impact * 100) if total_impact > 0 else 0
            
            explanations.append({
                'feature': feature_name,
                'impact': impact,
                'impact_percentage': round(impact_percentage, 1),  # Runde auf eine Nachkommastelle
                'description':get_feature_description(feature_name, impact)
            })
        
        # Sortiere Erklärungen nach absolutem Einfluss
        explanations.sort(key=lambda x: abs(x['impact']), reverse=True)
        
        # Interpretation der Vorhersage und Empfehlungen
        if pred_value > 0.7:
            status = "sehr wahrscheinlich wechselbereit"
            recommendations = [
                "Der Kandidat zeigt starke Anzeichen für einen bevorstehenden Wechsel.",
                "Aktive Ansprache empfohlen.",
                f"Haupteinflussfaktoren:",
                *[exp['description'] for exp in explanations]
            ]
        elif pred_value > 0.5:
            status = "wahrscheinlich wechselbereit"
            recommendations = [
                "Der Kandidat könnte für einen Wechsel offen sein.",
                "Regelmäßige Kontaktaufnahme empfohlen.",
                f"Haupteinflussfaktoren:",
                *[exp['description'] for exp in explanations]
            ]
        elif pred_value > 0.3:
            status = "möglicherweise wechselbereit"
            recommendations = [
                "Der Kandidat zeigt keine klaren Anzeichen für einen Wechsel.",
                "Beobachtung der Situation empfohlen.",
                f"Haupteinflussfaktoren:",
                *[exp['description'] for exp in explanations]
            ]
        else:
            status = "bleibt wahrscheinlich"
            recommendations = [
                "Der Kandidat zeigt wenig Interesse an einem Wechsel.",
                "Längerfristige Beziehungspflege empfohlen.",
                f"Haupteinflussfaktoren:",
                *[exp['description'] for exp in explanations]
            ]
        
        return {
            "confidence": [pred_value],
            "recommendations": recommendations,
            "explanations": explanations,
            "status": status
        }

        
    except Exception as e:
        print(f"Fehler bei der Vorhersage: {str(e)}")
        traceback.print_exc()  # Füge Stack Trace hinzu
        return {
            "confidence": [0.0], 
            "recommendations": [f"Fehler bei der Vorhersage: {str(e)}"],
            "explanations": []
        }

def get_feature_description(feature, impact):
    """Generiert beschreibende Texte für die Features basierend auf ihrem Einfluss"""
    if feature == 'Beschäftigungsdauer':
        if impact > 0:
            return "Längere Beschäftigungsdauer erhöht die Wechselwahrscheinlichkeit"
        return "Kürzere Beschäftigungsdauer verringert die Wechselwahrscheinlichkeit"
    
    elif feature == 'Positionslevel':
        if impact > 0:
            return "Höheres Positionslevel deutet auf erhöhte Wechselbereitschaft hin"
        return "Niedrigeres Positionslevel deutet auf geringere Wechselbereitschaft hin"
    
    elif feature == 'Branchenkontinuität':
        if impact > 0:
            return "Häufige Branchenwechsel erhöhen die Wechselwahrscheinlichkeit"
        return "Kontinuität in der Branche verringert die Wechselwahrscheinlichkeit"
    
    return "Dieser Faktor beeinflusst die Vorhersage"