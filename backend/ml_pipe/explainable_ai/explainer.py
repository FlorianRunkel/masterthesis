import torch
import numpy as np
import shap
import lime
from lime import lime_tabular
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

class ModelExplainer:
    def __init__(self, model, feature_names: List[str], model_type: str = "gru"):

        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        # eval() nur aufrufen, wenn vorhanden (z.B. bei PyTorch, nicht bei XGBoost)
        if hasattr(self.model, "eval"):
            self.model.eval()
    
    def create_model_wrapper(self, method: str = "shap"):
        if method == "shap":
            return self._create_shap_wrapper()
        elif method == "lime":
            return self._create_lime_wrapper()
        else:
            raise ValueError(f"Unbekannte Methode: {method}")
    
    def _create_shap_wrapper(self):
        """Erstellt einen SHAP-kompatiblen Wrapper."""
        if self.model_type == "gru":
            class GRUModelForSHAP(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x):
                    # Führe das GRU-Modell aus
                    # Das Modell gibt (output, attention_weights) zurück
                    output, _ = self.model(x)
                    return output
            
            return GRUModelForSHAP(self.model)
        
        elif self.model_type == "tft":
            # TFT-Modelle sind bereits SHAP-kompatibel
            return self.model
        
        elif self.model_type == "xgboost":
            # XGBoost-Modelle können direkt an TreeExplainer übergeben werden
            return self.model
        
        else:
            return self.model
    
    def _create_lime_wrapper(self):
        """Erstellt einen LIME-kompatiblen Wrapper."""
        class ModelForLIME:
            def __init__(self, model, model_type):
                self.model = model
                self.model_type = model_type
                if hasattr(self.model, 'eval'):
                    self.model.eval()
            def predict(self, X):
                print("DEBUG: LIME predict called with", type(X), X.shape, X.dtype if hasattr(X, 'dtype') else None)
                if self.model_type == "xgboost":
                    proba = self.model.predict_proba(X)
                    result = proba[:, 1]
                    arr = np.array(result, dtype=np.float64).flatten().copy()
                    print("DEBUG: LIME XGBoost predict returns", type(arr), arr.shape, arr.dtype, arr[:5])
                    return arr
                elif self.model_type == "gru":
                    preds = []
                    with torch.no_grad():
                        for i in range(X.shape[0]):
                            x_tensor = torch.tensor(X[i:i+1], dtype=torch.float32).unsqueeze(0)
                            output, _ = self.model(x_tensor)
                            preds.append(output.item() if hasattr(output, 'item') else float(output))
                    return np.array(preds)
                elif self.model_type == "tft":
                    if isinstance(X, np.ndarray):
                        return np.mean(X, axis=1)
                    else:
                        x_tensor = torch.tensor(X, dtype=torch.float32)
                        result = torch.mean(x_tensor, dim=1)
                        if hasattr(result, 'numpy'):
                            result = result.numpy()
                        return result
                else:
                    preds = []
                    with torch.no_grad():
                        for i in range(X.shape[0]):
                            x_tensor = torch.tensor(X[i:i+1], dtype=torch.float32)
                            output = self.model(x_tensor)
                            preds.append(output.item() if hasattr(output, 'item') else float(output))
                    return np.array(preds)
        return ModelForLIME(self.model, self.model_type)
    
    def calculate_shap_values(self, input_data, background_data=None, check_additivity=False):
        """
        Berechnet SHAP-Werte mit der Standard SHAP-Bibliothek.
        
        Args:
            input_data: Eingabedaten für das Modell
            background_data: Hintergrunddaten für SHAP
            check_additivity: SHAP-Additivitätspruefung
        """
        try:
            if self.model_type == "gru":
                return self._calculate_gru_shap_tabular(input_data)
            elif self.model_type == "xgboost":
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(input_data)
                return shap_values
            else:
                shap_model = self.create_model_wrapper("shap")
                if background_data is None:
                    background_data = torch.zeros((10, input_data.shape[1]))
                explainer = shap.DeepExplainer(shap_model, background_data)
                shap_values = explainer.shap_values(input_data, check_additivity=check_additivity)
                return shap_values
        except Exception as e:
            print(f"Fehler bei SHAP-Berechnung: {str(e)}")
            return None
    
    def _calculate_gru_shap_tabular(self, input_data):
        """
        Berechnet SHAP-Werte für GRU-Modelle mit KernelExplainer.
        """
        # Extrahiere Features aus dem Input-Tensor
        features = input_data.squeeze().numpy()  # (seq_len, features)
        
        # Verwende nur den letzten Zeitpunkt für SHAP
        last_features = features[-1] if len(features.shape) > 1 else features
        
        # Erstelle Hintergrunddaten
        background_data = np.random.normal(
            loc=last_features.mean(),
            scale=last_features.std(),
            size=(100, len(last_features))
        )
        
        # Erstelle einen Wrapper für die Vorhersage
        def predict_wrapper(X):
            predictions = []
            for i in range(X.shape[0]):
                # Konvertiere zu GRU-Input-Format
                gru_input = torch.tensor(X[i:i+1], dtype=torch.float32).unsqueeze(0)  # (1, 1, features)
                
                with torch.no_grad():
                    output, _ = self.model(gru_input)
                    predictions.append(output.item())
            return np.array(predictions)
        
        # Verwende KernelExplainer
        explainer = shap.KernelExplainer(
            predict_wrapper,
            background_data,
            feature_names=self.feature_names
        )
        
        # Berechne SHAP-Werte
        shap_values = explainer.shap_values(last_features.reshape(1, -1))
        
        return shap_values
    
    def _calculate_gru_shap_values(self, input_data):
        """
        Berechnet SHAP-Werte für GRU-Modelle durch Feature-Perturbation.
        """
        self.model.eval()
        
        # Extrahiere die Features aus dem Input-Tensor
        features = input_data.squeeze().numpy()  # (seq_len, features)
        
        # Berechne die Baseline-Vorhersage
        with torch.no_grad():
            baseline_pred, _ = self.model(input_data)
            baseline_value = baseline_pred.item()
        
        # Berechne SHAP-Werte für jedes Feature durch Perturbation
        shap_values = np.zeros(features.shape[1])  # Ein Wert pro Feature
        
        for feature_idx in range(features.shape[1]):
            # Erstelle eine Kopie der Features
            perturbed_features = features.copy()
            
            # Setze das aktuelle Feature auf 0 (Perturbation)
            perturbed_features[:, feature_idx] = 0
            
            # Konvertiere zurück zu Tensor
            perturbed_tensor = torch.tensor(perturbed_features, dtype=torch.float32).unsqueeze(0)
            
            # Berechne Vorhersage mit perturbiertem Feature
            with torch.no_grad():
                perturbed_pred, _ = self.model(perturbed_tensor)
                perturbed_value = perturbed_pred.item()
            
            # SHAP-Wert ist die Differenz zwischen Baseline und perturbierter Vorhersage
            shap_values[feature_idx] = baseline_value - perturbed_value
        
        return shap_values
    
    def calculate_lime_explanations(self, input_data, num_samples=1000, num_features=None, **kwargs):
        """
        Berechnet LIME-Erklärungen.
        
        Args:
            input_data: Eingabedaten für das Modell
            num_samples: Anzahl der LIME-Samples
            num_features: Anzahl der zu erklärenden Features
            **kwargs: Zusätzliche Parameter für modell-spezifische Implementierungen
        """
        if self.model_type == "tft":
            print("LIME wird für TFT-Modelle nicht unterstützt. Verwende nur SHAP-Erklärungen.")
            return None
        else:
            return self._calculate_standard_lime_explanations(input_data, num_samples, num_features)
    
    def _calculate_standard_lime_explanations(self, input_data, num_samples=1000, num_features=None):
        """Standard LIME-Berechnung für GRU und andere Modelle."""
        try:
            lime_model = self.create_model_wrapper("lime")
            print("DEBUG: input_data type", type(input_data))
            # Features als 2D-Array für LIME vorbereiten
            # Behandle sowohl numpy-Arrays als auch Torch-Tensoren
            if hasattr(input_data, 'numpy'):
                # Torch-Tensor
                features_2d = input_data.squeeze().numpy().reshape(1, -1)
            else:
                # Numpy-Array
                features_2d = input_data.squeeze().reshape(1, -1)
            print("DEBUG: features_2d", type(features_2d), features_2d.shape, features_2d.dtype)
            # Hintergrunddaten für LIME erstellen
            background_data = np.random.normal(
                loc=features_2d.mean(), 
                scale=features_2d.std(), 
                size=(100, features_2d.shape[1])
            )
            print("DEBUG: background_data", type(background_data), background_data.shape, background_data.dtype)
            # LIME-Explainer erstellen
            explainer = lime_tabular.LimeTabularExplainer(
                background_data,
                feature_names=self.feature_names,
                mode='regression',
                random_state=42,
                feature_selection='none'
            )
            print("DEBUG: LIME explainer created")

            # Modelltyp-spezifische predict-Funktion
            if self.model_type == "xgboost":
                def predict_fn(X):
                    return self.model.predict_proba(X)[:, 1]
            elif self.model_type == "gru":
                def predict_fn(X):
                    preds = []
                    with torch.no_grad():
                        for i in range(X.shape[0]):
                            x_tensor = torch.tensor(X[i:i+1], dtype=torch.float32).unsqueeze(0)
                            output, _ = self.model(x_tensor)
                            preds.append(output.item() if hasattr(output, 'item') else float(output))
                    return np.array(preds)
            else:
                # Fallback für andere Modelle
                def predict_fn(X):
                    return self.model.predict(X)

            print("DEBUG: calling explain_instance ...")
            explanation = explainer.explain_instance(
                features_2d[0], 
                predict_fn,
                num_features=num_features or len(self.feature_names),
                num_samples=num_samples
            )
            print("DEBUG: explain_instance finished")
            return explanation
        except Exception as e:
            print(f"Fehler bei Standard LIME-Berechnung: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None
    
    def extract_lime_results(self, explanation, min_impact=0.001):
        """
        Extrahiert Ergebnisse aus LIME-Erklärung und normalisiert die Impacts auf 100%.
        """
        if explanation is None:
            return []
        
        # Sammle alle LIME-Erklärungen mit Feature-Zuordnung
        feature_impacts = {}
        
        for feature_condition, weight in explanation.as_list():
            if abs(weight) > min_impact:
                # Extrahiere Feature-Name aus der Bedingung
                feature_name = feature_condition.split(' <= ')[0].split(' > ')[0].strip()
                
                # Finde den korrekten Feature-Namen
                matching_feature = None
                for name in self.feature_names:
                    if name.lower() in feature_name.lower() or feature_name.lower() in name.lower():
                        matching_feature = name
                        break
                
                if matching_feature:
                    impact = abs(weight)
                    if matching_feature in feature_impacts:
                        feature_impacts[matching_feature] += impact
                    else:
                        feature_impacts[matching_feature] = impact
        
        # **Normalisierung auf 100%**
        total_impact = sum(feature_impacts.values())
        lime_explanations = []
        for feature, impact in feature_impacts.items():
            percent = (impact / total_impact * 100) if total_impact > 0 else 0.0
            lime_explanations.append({
                "feature": feature,
                "impact_percentage": float(percent),
                "method": "LIME",
                "description": self._get_feature_description(feature)
            })
        
        # Nach Impact sortieren
        lime_explanations.sort(key=lambda x: x["impact_percentage"], reverse=True)
        return lime_explanations
    
    def extract_shap_results(self, shap_values, min_impact=0.001):
        """
        Extrahiert Ergebnisse aus SHAP-Werten.
        
        Args:
            shap_values: SHAP-Werte
            min_impact: Minimaler Impact für Feature-Inklusion
        """
        if shap_values is None:
            return []
        
        # Standard SHAP-Verarbeitung für alle Modelltypen
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        # Stelle sicher, dass wir mit 1D-Arrays arbeiten
        if len(shap_values.shape) > 1:
            if shap_values.shape[0] == 1:
                shap_values = shap_values[0]  # Erste Zeile
            else:
                shap_values = np.mean(shap_values, axis=0)  # Durchschnitt über Samples
        
        # Normalisiere die SHAP-Werte
        abs_values = np.abs(shap_values)
        total = np.sum(abs_values)
        
        if total > 0:
            norm_shap = (abs_values / total) * 100
        else:
            norm_shap = np.zeros_like(abs_values)
        
        # Stelle sicher, dass wir die richtige Anzahl von Features haben
        expected_features = len(self.feature_names)
        if len(norm_shap) != expected_features:
            print(f"Warnung: SHAP-Werte haben {len(norm_shap)} Features, erwartet {expected_features}")
            # Padden oder trimmen auf die erwartete Anzahl
            if len(norm_shap) < expected_features:
                norm_shap = np.pad(norm_shap, (0, expected_features - len(norm_shap)), 'constant')
            else:
                norm_shap = norm_shap[:expected_features]
        
        # Erstelle Explanations für alle Features
        explanations = []
        for i, (name, val) in enumerate(zip(self.feature_names, norm_shap)):
            # Stelle sicher, dass val ein skalarer Wert ist
            if hasattr(val, '__iter__') and not isinstance(val, str):
                if len(val) == 1:
                    val_scalar = float(val[0])
                else:
                    val_scalar = float(np.mean(val))
            else:
                val_scalar = float(val)
                
            if val_scalar > min_impact:
                explanations.append({
                    "feature": name,
                    "impact_percentage": val_scalar,
                    "method": "SHAP",
                    "description": self._get_feature_description(name)
                })
        
        # Nach Impact sortieren
        explanations.sort(key=lambda x: x["impact_percentage"], reverse=True)
        return explanations
    
    def combine_explanations(self, shap_explanations, lime_explanations):
        """
        Kombiniert SHAP und LIME Erklärungen.
        
        Args:
            shap_explanations: SHAP-Erklärungen
            lime_explanations: LIME-Erklärungen
        """
        combined = {}
        
        # Füge SHAP-Erklärungen hinzu
        for exp in shap_explanations:
            combined[exp["feature"]] = {
                "feature": exp["feature"],
                "shap_impact": exp["impact_percentage"],
                "lime_impact": 0.0,
                "combined_impact": exp["impact_percentage"]
            }
        
        # Füge LIME-Erklärungen hinzu
        for exp in lime_explanations:
            if exp["feature"] in combined:
                combined[exp["feature"]]["lime_impact"] = exp["impact_percentage"]
                combined[exp["feature"]]["combined_impact"] = (
                    combined[exp["feature"]]["shap_impact"] + exp["impact_percentage"]
                ) / 2
            else:
                combined[exp["feature"]] = {
                    "feature": exp["feature"],
                    "shap_impact": 0.0,
                    "lime_impact": exp["impact_percentage"],
                    "combined_impact": exp["impact_percentage"]
                }
        
        # Konvertiere zu Liste und sortiere
        combined_list = list(combined.values())
        combined_list.sort(key=lambda x: x["combined_impact"], reverse=True)
        
        return combined_list
    
    def _get_feature_description(self, feature_name):
        """
        Gibt eine Beschreibung für ein Feature zurück.
        """
        # Importiere die Funktionen aus den entsprechenden Modulen
        if self.model_type == "gru":
            from ..models.gru.predict import get_feature_description
            return get_feature_description(feature_name)
        elif self.model_type == "tft":
            from ..models.tft.predict import get_feature_description
            return get_feature_description(feature_name)
        else:
            return "This feature influences the prediction."

    def create_summary(self, explanations, method="SHAP"):
        """
        Erstellt eine Zusammenfassung der Erklärungen.
        
        Args:
            explanations: Liste der Erklärungen
            method: Methode (SHAP, LIME, oder Combined)
        """
        if not explanations:
            return f"Keine {method}-Erklärung verfügbar."
        
        summary_parts = []
        for exp in explanations:
            if method == "Combined":
                impact = exp["combined_impact"]
                summary_parts.append(f"{exp['feature']}: {impact:.1f}% (SHAP: {exp['shap_impact']:.1f}%, LIME: {exp['lime_impact']:.1f}%)")
            else:
                impact = exp["impact_percentage"]
                summary_parts.append(f"{exp['feature']}: {impact:.1f}% ({method})")
        
        return f"{method} Feature-Einflüsse:\n" + "\n".join(summary_parts) 