from flask import Blueprint, request, jsonify
from ml_pipe.data.database.mongodb import MongoDb
import logging
from bson import ObjectId

candidates_bp = Blueprint('candidates_bp', __name__)

'''
Helper function to check if a candidate already exists in the database
'''
def candidate_exists(candidate, mongo_db, collection_name):
    uid = candidate.get('uid')
    if not uid:
        return False

    if candidate.get('linkedinProfile'):
        query = {'linkedinProfile': candidate['linkedinProfile'], 'uid': uid}
        res = mongo_db.get(query, collection_name)
        if res['statusCode'] == 200 and res['data']:
            return True

    if candidate.get('firstName') and candidate.get('lastName'):
        query = {'firstName': candidate['firstName'], 'lastName': candidate['lastName'], 'uid': uid}
        res = mongo_db.get(query, collection_name)
        if res['statusCode'] == 200 and res['data']:
            return True

    return False

'''
Get all candidates from the database
'''
@candidates_bp.route('/api/candidates', methods=['GET'])
def get_all_candidates():
    try:
        mongo_db = MongoDb()
        result = mongo_db.get_all('candidates')

        if result['statusCode'] != 200:
            return jsonify({'error': result.get('error', 'No candidates found')}), result['statusCode']

        return jsonify(result['data']), 200

    except Exception as e:
        logging.error(f"Error getting candidates: {str(e)}")
        return jsonify({'error': str(e)}), 500

'''
Delete a candidate by id
'''
@candidates_bp.route('/api/candidates/<candidate_id>', methods=['DELETE'])
def delete_candidate(candidate_id):
    try:
        uid = request.headers.get('X-User-Uid')
        if not uid:
            return jsonify({'error': 'No user uid provided!'}), 400

        mongo_db = MongoDb()
        result = mongo_db.delete({'_id': ObjectId(candidate_id), 'uid': uid}, 'candidates')

        if result['statusCode'] == 200:
            return jsonify({'message': 'Candidate successfully deleted'}), 200
        else:
            return jsonify({'error': result.get('error', 'Error deleting candidate')}), result['statusCode']

    except Exception as e:
        logging.error(f"Error deleting candidate: {str(e)}")
        return jsonify({'error': 'Internal server error: ' + str(e)}), 500

'''
Save candidates in database.
'''
@candidates_bp.route('/api/candidates', methods=['POST'])
def save_candidates():
    try:
        candidates = request.json
        uid = request.headers.get('X-User-Uid')

        if not uid:
            if isinstance(candidates, dict) and 'uid' in candidates:
                uid = candidates['uid']
                candidates = candidates.get('candidates', [])

        if not uid:
            return jsonify({'error': 'No user uid provided!'}), 400
        if not candidates:
            return jsonify({'error': 'No candidates to save found'}), 400

        mongo_db = MongoDb()
        if mongo_db.db is None:
            mongo_db.get_mongo_client()
            if mongo_db.db is None:
                return jsonify({'error': 'Error connecting to database'}), 500

        saved_count = 0
        skipped_count = 0
        for candidate in candidates:
            logging.info(f"Save candidates: {candidate}")
            candidate['uid'] = uid
            if not candidate_exists(candidate, mongo_db, 'candidates'):
                result = mongo_db.create(candidate, 'candidates')
            else:
                result = {'statusCode': 409, 'error': 'Candidate already exists.'}

            if result['statusCode'] == 200: 
                saved_count += 1
                logging.info(f"Candidate successfully saved: {candidate.get('linkedinProfile','')} (UID: {uid})")
            else:
                skipped_count += 1
                logging.info(f"Error saving candidate: {result['error']}")

        return jsonify({
            'message': 'Candidates successfully saved',
            'savedCount': saved_count,
            'skippedCount': skipped_count,
            'reasonSkipped': 'Duplicates based on LinkedIn profile URL'
        }), 201

    except Exception as e:
        logging.error(f"Error saving candidates: {str(e)}")
        return jsonify({'error': 'Interner Serverfehler: ' + str(e)}), 500 