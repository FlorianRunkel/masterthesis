from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Modell laden (Pfad ggf. anpassen)
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    'models', 'xgboost', 'saved_models', 'xgboost_model_20250616_164839.joblib'
)
model = joblib.load(MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(port=6000, debug=True) 