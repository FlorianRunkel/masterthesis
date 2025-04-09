import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score, classification_report
import numpy as np

class XGBoostModel:
    def __init__(self, params=None):
        self.params = params or {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "use_label_encoder": False
        }
        self.model = xgb.XGBClassifier(**self.params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_val, y_val, show_report=False):
        preds = self.model.predict(X_val)
        f1 = f1_score(y_val, preds)
        acc = accuracy_score(y_val, preds)
        print(f"F1 Score:     {f1:.4f}")
        print(f"Accuracy:     {acc:.4f}")
        if show_report:
            print("Klassifikationsbericht:")
            print(classification_report(y_val, preds))
        return f1

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
