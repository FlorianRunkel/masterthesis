import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, randint
import numpy as np

class XGBoostModel:
    def __init__(self, params=None):

        #Default Parameters for XGBoost
        default_params = {
            "objective": "binary:logistic",       # Klassifikation mit Wahrscheinlichkeiten
            "eval_metric": "logloss",             # Metrik für binäre Klassifikation
            "tree_method": "hist",                # Schneller, speichereffizienter Baumalgorithmus
            "enable_categorical": True,           
            "learning_rate": 0.03,                 # Solide Lernrate
            "max_depth": 4,                       # Gute Tiefe für generalisierende Modelle
            "min_child_weight": 10,                # Weniger konservativ, mehr Splits erlaubt
            "subsample": 0.8,                     # Random Sampling pro Baum (Overfitting-Schutz)
            "colsample_bytree": 0.8,              # Feature Sampling pro Baum
            "gamma": 2,                           # Kein Split-Penalty
            "reg_alpha": 0.5,                       # Keine L1-Regularisierung (kann aktiviert werden)
            "reg_lambda": 2,                      # Standardmäßige L2-Regularisierung
            "n_estimators": 500,                  # Genug Bäume für sinnvolle Lernkurve
            "random_state": 42                    # Reproduzierbarkeit
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

    def evaluate(self, X_val, y_val, show_report=False):
        print("[INFO] Evaluating model...")
        preds = self.model.predict(X_val)
        f1 = f1_score(y_val, preds)
        acc = accuracy_score(y_val, preds)
        print(f"F1 Score:     {f1:.4f}")
        print(f"Accuracy:     {acc:.4f}")
        if show_report:
            print("\nKlassifikationsbericht:")
            print(classification_report(y_val, preds))
        return f1

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
            'max_depth': randint(15, 50),                         # Tiefer für komplexe Muster
            'min_child_weight': randint(1, 30),                  # Feinere Regularisierung
            'subsample': uniform(0.4, 0.6),                      # 0.4 bis 1.0
            'colsample_bytree': uniform(0.4, 0.6),               # 0.4 bis 1.0
            'eta': uniform(0.01, 0.29),                          # Feinere Lernraten
            'gamma': uniform(0, 5),                              # Pruning-Kontrolle
            'reg_alpha': uniform(0, 10),                         # L1-Regularisierung
            'reg_lambda': uniform(0, 15),                        # L2-Regularisierung
            'scale_pos_weight': [1, 2, 5, 10, 20, 50],           # Für Imbalance
            'n_estimators': randint(200, 2000),                  # Längeres Training
            'max_delta_step': randint(0, 10),                    # Für Class Imbalance
            'max_leaves': randint(10, 256),                      # Bei lossguide relevant
            'grow_policy': ['depthwise', 'lossguide'],
            'tree_method': ['hist'],                             # Optional: 'gpu_hist'
            'sampling_method': ['uniform', 'gradient_based']     # Sampling-Strategien
        }

        xgb_clf = xgb.XGBClassifier(tree_method="hist", objective="binary:logistic", use_label_encoder=False)
        random_search = RandomizedSearchCV(
            xgb_clf, param_distributions=param_dist, n_iter=30, scoring='f1', cv=3, verbose=2, n_jobs=-1, random_state=42
        )
        random_search.fit(X_train, y_train)

        print("Beste Parameter:", random_search.best_params_)
        print("Bester F1-Score:", random_search.best_score_)

        self.params.update(random_search.best_params_)
        self.model = xgb.XGBClassifier(**self.params, num_boost_round=100)
        return random_search.best_params_