import torch
import numpy as np
from typing import List
from .shap_explainer import ShapExplainer
from .lime_explainer import LimeExplainer

'''
Wrapper for GRU models to make them SHAP-compatible
'''
class GRUModelForSHAP(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output, _ = self.model(x)
        return output

'''
Wrapper for models to make them LIME-compatible
'''
class ModelForLIME:

    def __init__(self, model, model_type):
        self.model = model
        self.model_type = model_type
        if hasattr(self.model, 'eval'):
            self.model.eval()

    '''
    Predict for lime
    '''
    def predict(self, X):

        if self.model_type == "xgboost": # predict for xgboost
            proba = self.model.predict_proba(X)
            result = proba[:, 1]
            arr = np.array(result, dtype=np.float64).flatten().copy()
            return arr

        elif self.model_type == "gru": # predict for gru
            preds = []
            with torch.no_grad():
                for i in range(X.shape[0]):
                    x_tensor = torch.tensor(X[i:i+1], dtype=torch.float32).unsqueeze(0)
                    output, _ = self.model(x_tensor)
                    preds.append(output.item() if hasattr(output, 'item') else float(output))
            return np.array(preds)

        elif self.model_type == "tft": # predict for tft
            if isinstance(X, np.ndarray):
                return np.mean(X, axis=1)
            else:
                x_tensor = torch.tensor(X, dtype=torch.float32)
                result = torch.mean(x_tensor, dim=1)
                if hasattr(result, 'numpy'):
                    result = result.numpy()
                return result

        else: # predict for other models
            preds = []
            with torch.no_grad():
                for i in range(X.shape[0]):
                    x_tensor = torch.tensor(X[i:i+1], dtype=torch.float32)
                    output = self.model(x_tensor)
                    preds.append(output.item() if hasattr(output, 'item') else float(output))
            return np.array(preds)

'''
Model explainer
'''
class ModelExplainer:
    '''
    Init model explainer
    '''
    def __init__(self, model, feature_names: List[str], model_type: str = "gru"):
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.shap_explainer = ShapExplainer(model, feature_names, model_type)
        self.lime_explainer = LimeExplainer(model, feature_names, model_type)
        if hasattr(self.model, "eval"):
            self.model.eval()

    '''
    Create model wrapper
    '''
    def create_model_wrapper(self, method: str = "shap"):
        if method == "shap":
            return self._create_shap_wrapper()
        elif method == "lime":
            return self._create_lime_wrapper()
        else:
            raise ValueError(f"Unbekannte Methode: {method}")

    '''
    Create shap wrapper
    '''
    def _create_shap_wrapper(self):
        if self.model_type == "gru":
            return GRUModelForSHAP(self.model)
        elif self.model_type == "tft":
            return self.model
        elif self.model_type == "xgboost":
            return self.model
        else:
            return self.model

    '''
    Create lime wrapper
    '''
    def _create_lime_wrapper(self):
        return ModelForLIME(self.model, self.model_type)

    '''
    Calculate shap values
    '''
    def calculate_shap_values(self, input_data, **kwargs):
        return self.shap_explainer.calculate_shap_values(input_data, **kwargs)

    '''
    Calculate lime explanations
    '''
    def calculate_lime_explanations(self, input_data, **kwargs):
        return self.lime_explainer.calculate_lime_explanations(input_data, **kwargs)

    '''
    Extract lime results
    '''
    def extract_lime_results(self, explanation, **kwargs):
        return self.lime_explainer.extract_lime_results(explanation, **kwargs)

    '''
    Extract shap results
    '''
    def extract_shap_results(self, shap_values, **kwargs):
        return self.shap_explainer.extract_shap_results(shap_values, **kwargs)

    '''
    Combine explanations
    '''
    def combine_explanations(self, shap_explanations, lime_explanations):
        combined = {}
        for exp in shap_explanations:
            combined[exp["feature"]] = {
                "feature": exp["feature"],
                "shap_impact": exp["impact_percentage"],
                "lime_impact": 0.0,
                "combined_impact": exp["impact_percentage"]
            }
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
        combined_list = list(combined.values())
        combined_list.sort(key=lambda x: x["combined_impact"], reverse=True)
        return combined_list

    '''
    Get feature description
    '''
    def _get_feature_description(self, feature_name):
        if self.model_type == "gru":
            from ..models.gru.predict import get_feature_description
            return get_feature_description(feature_name)
        elif self.model_type == "tft":
            from ..models.tft.predict import get_feature_description
            return get_feature_description(feature_name)
        else:
            return "This feature influences the prediction."

    '''
    Create summary
    '''
    def create_summary(self, explanations, method="SHAP"):
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