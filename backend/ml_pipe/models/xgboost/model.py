import xgboost as xgb
from sklearn.metrics import f1_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import uniform, randint

class XGBoostModel:
    def __init__(self, params=None):
        default_params = {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "max_depth": 6,
            "min_child_weight": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eta": 0.1,
            "eval_metric": "logloss",
            "n_estimators": 100,
            "random_state": 42
        }
        self.params = params or default_params
        self.model = xgb.XGBClassifier(**self.params, num_boost_round=100)


    def train(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=10):
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
            verbose=False
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
            'max_depth': randint(3, 15),
            'min_child_weight': randint(1, 20),
            'subsample': uniform(0.5, 0.5),
            'colsample_bytree': uniform(0.5, 0.5),
            'eta': uniform(0.01, 0.19),
            'gamma': uniform(0, 1),
            'reg_alpha': uniform(0, 5),
            'reg_lambda': uniform(0.5, 9.5),
            'scale_pos_weight': [1, 2, 5, 10],
            'n_estimators': randint(100, 400),
            'max_delta_step': randint(0, 11),
            'max_leaves': randint(0, 51),
            'grow_policy': ['depthwise', 'lossguide']
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