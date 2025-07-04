import torch
import numpy as np
from lime import lime_tabular
from typing import List

class LimeExplainer:
    def __init__(self, model, feature_names: List[str], model_type: str = "gru"):
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        if hasattr(self.model, "eval"):
            self.model.eval()

    def calculate_lime_explanations(self, input_data, num_samples=1000, num_features=None, **kwargs):
        if self.model_type == "tft":
            print("LIME wird für TFT-Modelle nicht unterstützt. Verwende nur SHAP-Erklärungen.")
            return None
        else:
            return self._calculate_standard_lime_explanations(input_data, num_samples, num_features)

    def _calculate_standard_lime_explanations(self, input_data, num_samples=1000, num_features=None):
        try:
            print("DEBUG: input_data type", type(input_data))
            if hasattr(input_data, 'numpy'):
                features_2d = input_data.squeeze().numpy().reshape(1, -1)
            else:
                features_2d = input_data.squeeze().reshape(1, -1)
            print("DEBUG: features_2d", type(features_2d), features_2d.shape, features_2d.dtype)
            background_data = np.random.normal(
                loc=features_2d.mean(), 
                scale=features_2d.std(), 
                size=(100, features_2d.shape[1])
            )
            print("DEBUG: background_data", type(background_data), background_data.shape, background_data.dtype)
            explainer = lime_tabular.LimeTabularExplainer(
                background_data,
                feature_names=self.feature_names,
                mode='regression',
                random_state=42,
                feature_selection='none'
            )
            print("DEBUG: LIME explainer created")
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
        if explanation is None:
            return []
        feature_impacts = {}
        for feature_condition, weight in explanation.as_list():
            if abs(weight) > min_impact:
                feature_name = feature_condition.split(' <= ')[0].split(' > ')[0].strip()
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
        total_impact = sum(feature_impacts.values())
        lime_explanations = []
        for feature, impact in feature_impacts.items():
            percent = (impact / total_impact * 100) if total_impact > 0 else 0.0
            lime_explanations.append({
                "feature": feature,
                "impact_percentage": float(percent),
                "method": "LIME",
                "description": "Dieses Merkmal beeinflusst die Vorhersage."
            })
        lime_explanations.sort(key=lambda x: x["impact_percentage"], reverse=True)
        return lime_explanations 