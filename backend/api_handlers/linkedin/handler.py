from flask import Blueprint, request, jsonify
from api_handlers.linkedin.unipilelayer import UnipileLayer
import logging
import json
linkedin_bp = Blueprint('linkedin_bp', __name__)

API_KEY = "9JJfa48E.0/iK1EJuHNQF5DQc1S8Mw99LnGvBvFZV/ZRxloy4+7A="
ACCOUNT_ID = "_7px0TZeTYWzpIB-tj7n1Q"
SUBDOMAIN = "api3"
PORT = "13342"

unipile = UnipileLayer(API_KEY, SUBDOMAIN, PORT)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

'''
Scrape LinkedIn profile
'''
@linkedin_bp.route('/api/scrape-linkedin', methods=['POST'])
def scrape_linkedin():
    try:
        data = request.get_json()
        linkedin_url = data.get('url')
        if not linkedin_url:
            return jsonify({'error': 'No LinkedIn URL provided'}), 400

        profile_raw = unipile.get_profile_by_url(linkedin_url, ACCOUNT_ID)
        if not profile_raw:
            return jsonify({'error': 'Profile could not be fetched'}), 500

        logger.error(f"Unipile Profile: {str(profile_raw)}")

        model_input = unipile.transform_profile(profile_raw)
        logger.error(f"Unipile Model Input: {str(model_input)}")

        profile_info = json.loads(model_input['linkedinProfileInformation'])

        frontend_profile = {
            'name': model_input['name'],
            'firstName': model_input['firstName'],
            'lastName': model_input['lastName'],
            'currentTitle': profile_info.get('headline', ''),
            'location': profile_info.get('location', ''),
            'imageUrl': profile_raw.get('profile_picture_url', ''),
            'industry': profile_raw.get('industry', ''),
            'summary': profile_raw.get('summary', ''),
            'experience': [
                {
                    'title': exp.get('position', ''),
                    'company': exp.get('company', ''),
                    'startDate': exp.get('startDate', ''),
                    'endDate': exp.get('endDate', ''),
                    'location': exp.get('location', ''),
                    'description': exp.get('positionDescription', '')
                }
                for exp in profile_info.get('workExperience', [])
            ],
            'education': [
                {
                    'degree': edu.get('degree', ''),
                    'school': edu.get('institution', ''),
                    'startDate': edu.get('startDate', ''),
                    'endDate': edu.get('endDate', '')
                }
                for edu in profile_info.get('education', [])
            ],
            'skills': profile_info.get('skills', [])
        }

        # Combine both formats
        response_data = {
            **model_input,
            **frontend_profile
        }

        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Unipile Error: {str(e)}")
        return jsonify({'error': str(e)}), 500 