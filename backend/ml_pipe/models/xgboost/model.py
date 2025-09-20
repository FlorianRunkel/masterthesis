import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from scipy.stats import uniform, randint

'''
XGBoost Model
'''
class XGBoostModel:
    def __init__(self, params=None):
        default_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "learning_rate": 0.03,
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

        self.params = params if params is not None else default_params
        self.model = xgb.XGBClassifier(**self.params)

    def encode_categoricals(self, X):
        X_encoded = X.copy()
        for col in X_encoded.select_dtypes(include=['object', 'category']).columns:
            X_encoded[col] = LabelEncoder().fit_transform(X_encoded[col].astype(str))
        return X_encoded

    def train(self, X_train, y_train, X_val=None, y_val=None):
        if hasattr(X_train, 'isnull') and X_train.isnull().any().any():
            raise ValueError("Training data contains NaN values.")

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]

        if eval_set:
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=False,
                early_stopping_rounds=50
            )
        else:
            self.model.fit(
                X_train,
                y_train,
                verbose=False
            )

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        print("Evaluation results:")
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
        }

        xgb_clf = xgb.XGBClassifier(tree_method="hist", objective="binary:logistic", eval_metric="logloss")
        grid_search = GridSearchCV(xgb_clf, param_grid, scoring='f1', cv=3, verbose=2, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        print("Best parameters:", grid_search.best_params_)
        print("Best F1-Score:", grid_search.best_score_)

        self.params.update(grid_search.best_params_)
        self.model = xgb.XGBClassifier(**self.params)
        return grid_search.best_params_

    def randomized_search(self, X_train, y_train):
        # Categorical Encoding
        X_train_encoded = self.encode_categoricals(X_train)

        param_dist = {
            'max_depth': randint(4, 12),
            'min_child_weight': randint(1, 20),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'learning_rate': uniform(0.001, 0.2),
            'gamma': uniform(0, 5),
            'reg_alpha': uniform(0, 5),
            'reg_lambda': uniform(0, 10),
            'n_estimators': randint(200, 800),
            'max_delta_step': randint(0, 10),
            'tree_method': ['hist'],
        }

        xgb_clf = xgb.XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
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

        print("Start randomized search...")
        random_search.fit(X_train_encoded, y_train)

        print("\nBest parameters:", random_search.best_params_)
        print("Best F1-Score:", random_search.best_score_)

        self.params.update(random_search.best_params_)
        self.model = xgb.XGBClassifier(**self.params)

        return random_search.best_params_