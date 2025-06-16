import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, randint
import numpy as np

class XGBoostModel:
    def __init__(self, params=None):

        #Default Parameters for XGBoost
        default_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "enable_categorical": True,
            "learning_rate": 0.03,          # nicht zu niedrig
            "max_depth": 6,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.5,
            "reg_alpha": 0.1,
            "reg_lambda": 1,
            "n_estimators": 500,
            "random_state": 42
        }
        '''
        default_params = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "grow_policy": "depthwise",
            "max_depth": 8,
            "max_leaves": 151,
            "min_child_weight": 21,
            "subsample": 0.79,
            "learning_rate": 0.05,
            "eval_metric": "logloss",
            "n_estimators": 1015,
            "random_state": 42,
            "num_boost_round": 1000,
            'max_delta_step': 3,
            'max_leaves': 151,
            'reg_alpha': 7.464914051180242,
            'reg_lambda': 9.74449348570822,
            'sampling_method': 'uniform',
            'tree_method': 'hist',
            'colsample_bytree': 0.815137118615616,
            'gamma': 1.121346547302799,
        }
        '''
        self.params = params or default_params
        self.model = xgb.XGBClassifier(**self.params)

    def train(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=50):
        print("[INFO] Training XGBoost model...")

        if hasattr(X_train, 'isnull') and X_train.isnull().any().any():
            raise ValueError("Trainingsdaten enthalten NaN-Werte.")

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )

        print("[INFO] Training completed.")


    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        print("Evaluationsergebnisse:")
        print(f"Accuracy:  {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall:    {rec:.3f}")
        print(f"F1-Score:  {f1:.3f}")
        print(f"ROC AUC:   {auc:.3f}")
        print("Confusion Matrix:\n", cm)

        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def plot_feature_importance(self, max_num_features=10):
        print("[INFO] Plotting feature importance...")
        xgb.plot_importance(self.model, max_num_features=max_num_features)
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.show()

    def grid_search(self, X_train, y_train):
        param_grid = {
            'max_depth': [4, 8],
            'min_child_weight': [1, 10],
            'subsample': [0.7, 1.0],
            'colsample_bytree': [0.7, 1.0],
            'eta': [0.05, 0.2],
            'gamma': [0, 0.2],
            'reg_alpha': [0, 1],
            'reg_lambda': [1, 5],
            'n_estimators': [100, 200],
            'grow_policy': ['depthwise', 'lossguide']
        }

        xgb_clf = xgb.XGBClassifier(tree_method="hist", objective="binary:logistic", use_label_encoder=False)
        grid_search = GridSearchCV(xgb_clf, param_grid, scoring='f1', cv=3, verbose=2, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        print("Beste Parameter:", grid_search.best_params_)
        print("Bester F1-Score:", grid_search.best_score_)

        self.params.update(grid_search.best_params_)
        self.model = xgb.XGBClassifier(**self.params, num_boost_round=100)
        return grid_search.best_params_

    def randomized_search(self, X_train, y_train):

        param_dist = {
            'max_depth': randint(4, 12),
            'min_child_weight': randint(1, 20),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'learning_rate': uniform(0.001, 0.2),  # war vorher fälschlich `eta`
            'gamma': uniform(0, 5),
            'reg_alpha': uniform(0, 5),
            'reg_lambda': uniform(0, 10),
            'n_estimators': randint(200, 800),
            'max_delta_step': randint(0, 10),
            'max_leaves': randint(10, 128),
            'grow_policy': ['depthwise', 'lossguide'],
            'tree_method': ['hist'],
            'sampling_method': ['uniform', 'gradient_based']
        }

        xgb_clf = xgb.XGBClassifier(
            objective="binary:logistic",
            enable_categorical=True,
            tree_method="hist",
            use_label_encoder=False,  # wird ggf. ignoriert, aber harmlos
            eval_metric="logloss",
            random_state=42
        )

        random_search = RandomizedSearchCV(
            xgb_clf,
            param_distributions=param_dist,
            n_iter=30,
            scoring='f1',
            cv=3,
            verbose=2,
            n_jobs=-1,
            random_state=42
        )

        print("[INFO] Starte Randomized Search...")
        random_search.fit(X_train, y_train)

        print("\nBeste Parameter:", random_search.best_params_)
        print("Bester F1-Score :", random_search.best_score_)

        self.params.update(random_search.best_params_)
        self.model = xgb.XGBClassifier(**self.params)
        return random_search.best_params_