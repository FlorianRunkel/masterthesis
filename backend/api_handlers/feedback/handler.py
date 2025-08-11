from flask import Blueprint, request, jsonify
from ml_pipe.data.database.mongodb import MongoDb
import logging
from datetime import datetime

feedback_bp = Blueprint('feedback_bp', __name__)

'''
Save feedback in database
'''
@feedback_bp.route('/api/feedback', methods=['POST'])
def save_feedback():
    try:
        data = request.get_json()
        uid = request.headers.get('X-User-Uid')
        if not uid:
            return jsonify({'error': 'No user uid provided!'}), 400
        feedback = {
            'uid': uid,
            'freeText': data.get('freeText', ''),
            'prognoseBewertung': data.get('prognoseBewertung', []),
            'bewertungsskala': data.get('bewertungsskala', []),
            'explanationFeedback': data.get('explanationFeedback', {}),
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        mongo_db = MongoDb()
        result = mongo_db.create(feedback, 'feedback')
        if result['statusCode'] == 200:
            return jsonify({'message': 'Feedback saved!'}), 201
        else:
            return jsonify({'error': result.get('error', 'Error saving feedback')}), 500
    except Exception as e:
        logging.error(f"Error saving feedback: {str(e)}")
        return jsonify({'error': str(e)}), 500

'''
Get all feedback from the database
'''
@feedback_bp.route('/api/feedback', methods=['GET'])
def get_feedback():
    try:
        mongo_db = MongoDb()
        result = mongo_db.get_all('feedback')
        if result['statusCode'] == 200:
            return jsonify(result['data']), 200
        else:
            return jsonify({'error': result.get('error', 'Error getting feedback')}), 500
    except Exception as e:
        logging.error(f"Error getting feedback: {str(e)}")
        return jsonify({'error': str(e)}), 500 