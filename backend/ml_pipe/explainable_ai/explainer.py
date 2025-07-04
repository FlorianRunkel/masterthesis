import torch
import numpy as np  
from typing import List
from .shap_explainer import ShapExplainer
from .lime_explainer import LimeExplainer

class ModelExplainer:
    def __init__(self, model, feature_names: List[str], model_type: str = "gru"):
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.shap_explainer = ShapExplainer(model, feature_names, model_type)
        self.lime_explainer = LimeExplainer(model, feature_names, model_type)
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
    
    def calculate_shap_values(self, input_data, **kwargs):
        return self.shap_explainer.calculate_shap_values(input_data, **kwargs)
    
    def calculate_lime_explanations(self, input_data, **kwargs):
        return self.lime_explainer.calculate_lime_explanations(input_data, **kwargs)
    
    def extract_lime_results(self, explanation, **kwargs):
        return self.lime_explainer.extract_lime_results(explanation, **kwargs)
    
    def extract_shap_results(self, shap_values, **kwargs):
        return self.shap_explainer.extract_shap_results(shap_values, **kwargs)
    
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