from flask import Blueprint, request, jsonify
import pandas as pd
import logging
import json
import re

prediction_bp = Blueprint('prediction_bp', __name__)

'''
Helper function to preprocess dates and time
'''
def preprocess_dates_time(data):
    def to_mm_yyyy(date_str):
        if not date_str or date_str == "Present":
            return "Present"
        if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
            year, month, _ = date_str.split('-')
            return f"{month}/{year}"
        if re.match(r"^\d{4}$", date_str):
            return f"01/{date_str}"
        if re.match(r"^\d{2}/\d{4}$", date_str):
            return date_str
        return date_str

    for exp_key in ['experience', 'workExperience']:
        for exp in data.get(exp_key, []):
            exp['startDate'] = to_mm_yyyy(exp.get('startDate', ''))
            exp['endDate'] = to_mm_yyyy(exp.get('endDate', ''))
    for edu in data.get('education', []):
        edu['startDate'] = to_mm_yyyy(edu.get('startDate', ''))
        edu['endDate'] = to_mm_yyyy(edu.get('endDate', ''))
    return data

'''
Predict career for a single profile
'''
@prediction_bp.route('/predict', methods=['POST'])
def predict_career():
    try:
        data = request.get_json()
        logging.info(f"Incoming data: {data}")

        if "linkedinProfileInformation" in data:
            try:
                profile_data = json.loads(data["linkedinProfileInformation"])
            except Exception as json_err:
                logging.error(f"JSON parsing error: {str(json_err)}")
                return jsonify({'error': f'Invalid JSON format: {str(json_err)}'}), 400
        else:
            profile_data = data

        model_type = data.get('modelType', 'tft').lower()
        logging.info(f"Use model: {model_type}")

        # Get preloaded models from global cache
        from app import loaded_models
        
        if model_type not in loaded_models:
            logging.error(f"Model {model_type} not found in preloaded models")
            return jsonify({'error': f"Model {model_type} not available"}), 400

        preloaded_model = loaded_models[model_type]
        logging.info(f"Using preloaded {model_type} model")

        # Import prediction modules
        model_predictors = {
            "gru": "ml_pipe.models.gru.predict",
            "xgboost": "ml_pipe.models.xgboost.predict",
            "tft": "ml_pipe.models.tft.predict"
        }

        module = __import__(model_predictors[model_type], fromlist=['predict'])
        logging.info(f"Module loaded successfully: {model_predictors[model_type]}")

        if model_type in ['gru', 'tft']:
            profile_data = preprocess_dates_time(profile_data)

        # Pass preloaded model to predict function
        prediction = module.predict(profile_data, preloaded_model=preloaded_model)

        if model_type == 'xgboost':
            confidence_list = prediction.get('confidence', [])
            confidence_value = max(0.0, confidence_list[0] if confidence_list else 0.0)
            
            recommendations_list = prediction.get('recommendations', [])
            recommendations_value = recommendations_list[0] if recommendations_list else "No recommendation available"
            
            formatted_prediction = {
                'confidence': confidence_value,
                'recommendations': recommendations_value,
                'status': prediction.get('status', ''),
                'explanations': prediction.get('explanations', []),
                'shap_explanations': prediction.get('shap_explanations', []),
                'lime_explanations': prediction.get('lime_explanations', []),
                'shap_summary': prediction.get('shap_summary', ''),
                'lime_summary': prediction.get('lime_summary', ''),
                'llm_explanation': prediction.get('llm_explanation', '')
            }
        else:
            formatted_prediction = {
                'confidence': prediction['confidence'],
                'recommendations': prediction['recommendations'],
                'status': prediction.get('status', ''),
                'shap_explanations': prediction.get('shap_explanations', []),
                'lime_explanations': prediction.get('lime_explanations', []),
                'shap_summary': prediction.get('shap_summary', ''),
                'lime_summary': prediction.get('lime_summary', ''),
                'llm_explanation': prediction.get('llm_explanation', '')
            }

        logging.info(f"Formatted Prediction: {formatted_prediction}")
        return jsonify(formatted_prediction)

    except Exception as e:
        logging.error(f"Fehler bei der Vorhersage: {str(e)}")
        return jsonify({'error': str(e)}), 500

'''
Predict career for a batch of profiles
'''
@prediction_bp.route('/predict-batch', methods=['POST'])
def predict_batch():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file found in request'}), 400

        file = request.files['file']
        logging.info(f"Received file: {file.filename}")

        try:
            df = pd.read_csv(file)
            logging.info(f"CSV loaded successfully with {len(df)} rows")
        except Exception as csv_err:
            logging.error(f"CSV parsing error: {str(csv_err)}")
            return jsonify({'error': f'Error reading CSV file: {str(csv_err)}'}), 400

        try:
            model_type = request.form.get('modelType', 'xgboost').lower()
            
            # Get preloaded models from global cache
            from app import loaded_models
            
            if model_type not in loaded_models:
                logging.error(f"Model {model_type} not found in preloaded models")
                return jsonify({'error': f"Model {model_type} not available"}), 400

            preloaded_model = loaded_models[model_type]
            logging.info(f"Using preloaded {model_type} model for batch prediction")

            model_predictors = {
                "gru": "ml_pipe.models.gru.predict",
                "xgboost": "ml_pipe.models.xgboost.predict",
                "tft": "ml_pipe.models.tft.predict"
            }

            if model_type not in model_predictors:
                return jsonify({'error': f"Unknown model type: {model_type}"}), 400

            module = __import__(model_predictors[model_type], fromlist=['predict'])
            logging.info("Model module imported successfully")
        except Exception as import_err:
            logging.error(f"Model import error: {str(import_err)}")
            return jsonify({'error': f'Error loading model: {str(import_err)}'}), 500

        results = []
        for idx, row in df.iterrows():
            try:
                logging.info(f"Processing row {idx+1}/{len(df)}")

                if "linkedinProfileInformation" not in row or pd.isna(row["linkedinProfileInformation"]):
                    results.append({"firstName": row.get("firstName", ""),"lastName": row.get("lastName", ""),"linkedinProfile": row.get("profileLink", ""),"error": "Fehlende LinkedIn-Profilinformationen"})
                    continue

                try:
                    profile_data = json.loads(row["linkedinProfileInformation"])
                except Exception as json_err:
                    logging.error(f"JSON parsing error for row {idx+1}: {str(json_err)}")
                    results.append({"firstName": row.get("firstName", ""),"lastName": row.get("lastName", ""),"linkedinProfile": row.get("profileLink", ""),"error": f"Ungültiges JSON-Format: {str(json_err)}"})
                    continue

                if model_type == 'tft':
                    profile_data = preprocess_dates_time(profile_data)

                # Pass preloaded model to predict function
                prediction = module.predict(profile_data, preloaded_model=preloaded_model)

                if "error" in prediction:
                    results.append({"firstName": row.get("firstName", ""),"lastName": row.get("lastName", ""),"linkedinProfile": row.get("profileLink", ""),"error": prediction["error"]})
                else:
                    results.append({
                        "firstName": row.get("firstName", ""),
                        "lastName": row.get("lastName", ""),
                        "linkedinProfile": row.get("profileLink", ""),
                        "confidence": prediction["confidence"],
                        "recommendations": prediction["recommendations"],
                        "status": prediction.get("status", ""),
                        "explanations": prediction.get("explanations", []),
                        "shap_explanations": prediction.get("shap_explanations", []),
                        "lime_explanations": prediction.get("lime_explanations", []),
                        "shap_summary": prediction.get("shap_summary", ""),
                        "lime_summary": prediction.get("lime_summary", ""),
                        "llm_explanation": prediction.get("llm_explanation", "")
                    })

            except Exception as user_err:
                logging.error(f"Error processing row {idx+1}: {str(user_err)}")
                results.append({"firstName": row.get("firstName", ""),"lastName": row.get("lastName", ""),"linkedinProfile": row.get("profileLink", ""),"error": f"Fehler bei der Verarbeitung: {str(user_err)}"})

        logging.info(f"Processing completed. Successful: {sum(1 for r in results if 'error' not in r)}, Failed: {sum(1 for r in results if 'error' in r)}")
        return jsonify({"results": results})

    except Exception as e:
        logging.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500