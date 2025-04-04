import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ------------------------------
# 1. Dummy-Daten vorbereiten
# ------------------------------
def create_flat_dataset(num_samples=1000, seq_len=5, input_size=10):
    # 3D: [samples, time, features] → 2D flatten: [samples, time * features]
    X_seq = np.random.randn(num_samples, seq_len, input_size)
    X_flat = X_seq.reshape(num_samples, -1)
    y = np.random.randint(0, 2, size=(num_samples,))
    return X_flat, y

# ------------------------------
# 2. XGBoost Training
# ------------------------------
def train_xgboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", use_label_encoder=False)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")

    return model

# ------------------------------
# 3. Main
# ------------------------------
if __name__ == "__main__":
    X, y = create_flat_dataset(num_samples=1000, seq_len=5, input_size=10)
    trained_model = train_xgboost(X, y)