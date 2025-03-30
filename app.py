from flask import Flask, render_template, request, jsonify
from flask import Flask, request, jsonify
from models.groq.groq_predict import generate_groq_prediction
from models.custom_llm.predict_model import generate_custom_llm
import logging

app = Flask(__name__)
app.logger.setLevel(logging.INFO) 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extrahiere Nutzereingaben  
    first_name = data.get('firstName', '')
    last_name = data.get('lastName', '')    
    location = data.get('location', '')
    experiences = data.get('experiences', [])
    model_type = data.get('modelType', 'transformer') 

    app.logger.info(f"Data received: {data}")

    # Modell-Auswahl
    if model_type == "groq":
        prediction = generate_groq_prediction(first_name, last_name, location, experiences)  
    elif model_type == "customLLM":
        prediction = generate_custom_llm(first_name, last_name, location, experiences) 
    else:
        prediction = "Modell noch nicht integriert"

    return jsonify({"prediction": prediction})

if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5001, debug=True)