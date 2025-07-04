import torch
import numpy as np
import shap
from typing import List

class ShapExplainer:
    def __init__(self, model, feature_names: List[str], model_type: str = "gru"):
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        if hasattr(self.model, "eval"):
            self.model.eval()

    def calculate_shap_values(self, input_data, background_data=None, check_additivity=False):
        try:
            if self.model_type == "gru":
                return self._calculate_gru_shap_tabular(input_data)
            elif self.model_type == "xgboost":
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(input_data)
                return shap_values
            else:
                shap_model = self.model
                if background_data is None:
                    background_data = torch.zeros((10, input_data.shape[1]))
                explainer = shap.DeepExplainer(shap_model, background_data)
                shap_values = explainer.shap_values(input_data, check_additivity=check_additivity)
                return shap_values
        except Exception as e:
            print(f"Fehler bei SHAP-Berechnung: {str(e)}")
            return None

    def _calculate_gru_shap_tabular(self, input_data):
        features = input_data.squeeze().numpy()
        last_features = features[-1] if len(features.shape) > 1 else features
        background_data = np.random.normal(
            loc=last_features.mean(),
            scale=last_features.std(),
            size=(100, len(last_features))
        )
        def predict_wrapper(X):
            predictions = []
            for i in range(X.shape[0]):
                gru_input = torch.tensor(X[i:i+1], dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    output, _ = self.model(gru_input)
                    predictions.append(output.item())
            return np.array(predictions)
        explainer = shap.KernelExplainer(
            predict_wrapper,
            background_data,
            feature_names=self.feature_names
        )
        shap_values = explainer.shap_values(last_features.reshape(1, -1))
        return shap_values

    def extract_shap_results(self, shap_values, min_impact=0.001):
        if shap_values is None:
            return []
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        if len(shap_values.shape) > 1:
            if shap_values.shape[0] == 1:
                shap_values = shap_values[0]
            else:
                shap_values = np.mean(shap_values, axis=0)
        abs_values = np.abs(shap_values)
        total = np.sum(abs_values)
        if total > 0:
            norm_shap = (abs_values / total) * 100
        else:
            norm_shap = np.zeros_like(abs_values)
        expected_features = len(self.feature_names)
        if len(norm_shap) != expected_features:
            print(f"Warnung: SHAP-Werte haben {len(norm_shap)} Features, erwartet {expected_features}")
            if len(norm_shap) < expected_features:
                norm_shap = np.pad(norm_shap, (0, expected_features - len(norm_shap)), 'constant')
            else:
                norm_shap = norm_shap[:expected_features]
        explanations = []
        for i, (name, val) in enumerate(zip(self.feature_names, norm_shap)):
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
                    "description": "Dieses Merkmal beeinflusst die Vorhersage."
                })
        explanations.sort(key=lambda x: x["impact_percentage"], reverse=True)
        return explanations 