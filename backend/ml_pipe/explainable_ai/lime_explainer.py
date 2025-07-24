import torch
import numpy as np
from lime import lime_tabular
from typing import List

'''
LIME explainer
'''
class LimeExplainer:
    '''
    Init lime explainer
    '''
    def __init__(self, model, feature_names: List[str], model_type: str = "gru"):
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        if hasattr(self.model, "eval"):
            self.model.eval()

    '''
    Calculate lime explanations
    '''
    def calculate_lime_explanations(self, input_data, num_samples=1000, num_features=None, **kwargs):
        if self.model_type == "tft":
            print("LIME is not supported for TFT models. Use only SHAP explanations.")
            return None
        else:
            return self._calculate_standard_lime_explanations(input_data, num_samples, num_features)

    '''
    Predict for xgboost
    '''
    def _predict_xgboost(self, X):
        return self.model.predict_proba(X)[:, 1]

    '''
    Predict for gru
    '''
    def _predict_gru(self, X):
        preds = []
        with torch.no_grad():
            for i in range(X.shape[0]):
                x_tensor = torch.tensor(X[i:i+1], dtype=torch.float32).unsqueeze(0)
                output, _ = self.model(x_tensor)
                preds.append(output.item() if hasattr(output, 'item') else float(output))
        return np.array(preds)

    '''
    Predict for tft
    '''
    def _predict_tft(self, X):
        if isinstance(X, np.ndarray):
            return np.mean(X, axis=1)
        else:
            x_tensor = torch.tensor(X, dtype=torch.float32)
            result = torch.mean(x_tensor, dim=1)
            if hasattr(result, 'numpy'):
                result = result.numpy()
            return result

    '''
    Predict for other models
    '''
    def _predict_other(self, X):
        preds = []
        with torch.no_grad():
            for i in range(X.shape[0]):
                x_tensor = torch.tensor(X[i:i+1], dtype=torch.float32)
                output = self.model(x_tensor)
                preds.append(output.item() if hasattr(output, 'item') else float(output))
        return np.array(preds)

    '''
    Get predict function
    '''
    def _get_predict_fn(self):
        if self.model_type == "xgboost":
            return self._predict_xgboost
        elif self.model_type == "gru":
            return self._predict_gru
        elif self.model_type == "tft":
            return self._predict_tft
        else:
            return self._predict_other

    '''
    Calculate standard lime explanations
    '''
    def _calculate_standard_lime_explanations(self, input_data, num_samples=1000, num_features=None):
        try:
            if hasattr(input_data, 'numpy'):
                features_2d = input_data.squeeze().numpy().reshape(1, -1)
            else:
                features_2d = input_data.squeeze().reshape(1, -1)

            background_data = np.random.normal(
                loc=features_2d.mean(), 
                scale=features_2d.std(), 
                size=(100, features_2d.shape[1])
            )

            explainer = lime_tabular.LimeTabularExplainer(
                background_data,
                feature_names=self.feature_names,
                mode='regression',
                random_state=42,
                feature_selection='none'
            )

            predict_fn = self._get_predict_fn()
            explanation = explainer.explain_instance(
                features_2d[0], 
                predict_fn,
                num_features=num_features or len(self.feature_names),
                num_samples=num_samples
            )
            return explanation
        except Exception as e:
            print(f"Error calculating standard lime explanations: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None

    '''
    Extract lime results
    '''
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
                "description": "This feature influences the prediction."
            })
        lime_explanations.sort(key=lambda x: x["impact_percentage"], reverse=True)
        return lime_explanations 