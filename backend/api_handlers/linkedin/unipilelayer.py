import requests
import logging
import json
from datetime import datetime
from random import randint
from time import sleep
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

class UnipileLayer:
    def __init__(self, api_key, subdomain, port):
        self.api_key = api_key
        self.subdomain = subdomain
        self.port = port
        self.headers = {
            'X-API-KEY': self.api_key,
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }

    def get_profile_by_url(self, linkedin_url, account_id, api_type='classic'):
        # Extrahiere public identifier
        public_identifier = linkedin_url.rstrip('/').split('/in/')[1]
        if api_type == 'classic':
            url = f'https://{self.subdomain}.unipile.com:{self.port}/api/v1/users/{public_identifier}?linkedin_sections=%2A&account_id={account_id}'
        elif api_type == 'recruiter':
            url = f'https://{self.subdomain}.unipile.com:{self.port}/api/v1/users/{public_identifier}?linkedin_api=recruiter&linkedin_sections=%2A&account_id={account_id}'
        else:    
            raise ValueError('Unbekannter API-Typ')
        sleep(randint(1, 3))
        response = requests.get(url, headers=self.headers)
        if response.ok:
            return response.json()
        else:
            logger.error(f"Unipile API Fehler: {response.status_code} - {response.text}")
            return None
    
    def format_duration(self, start, end):
        def to_date_str(date_str):
            if not date_str or not isinstance(date_str, str):
                return ""
            parts = date_str.split("/")
            if len(parts) == 2:
                return f"{parts[1]}-{parts[0]}"
            elif len(parts) == 1:
                return date_str
            return ""

        start_fmt = to_date_str(start)
        end_fmt = to_date_str(end) if end else "Present"

        if start_fmt:
            return f"{start_fmt} - {end_fmt}"
        return ""
    
    def transform_profile(self, input_data):
        # LinkedIn Basisdaten
        first_name = input_data.get("first_name", "")
        last_name = input_data.get("last_name", "")
        public_identifier = input_data.get("public_identifier", "")
        profile_url = f"https://www.linkedin.com/in/{public_identifier}" if public_identifier else ""
        image_url = input_data.get("profile_picture_url", "")

        # Education
        education_list = []
        for edu in input_data.get("education", []):
            education_list.append({
                "degree": edu.get("degree", ""),
                "institution": edu.get("school", ""),
                "startDate": edu.get("start", ""),
                "endDate": edu.get("end", ""),
                "duration": self.format_duration(edu.get("start", ""), edu.get("end", ""))
            })

        # Work experience
        work_list = []
        for exp in input_data.get("work_experience", []):
            work_list.append({
                "position": exp.get("position", ""),
                "company": exp.get("company", ""),
                "location": exp.get("location", ""),
                "startDate": exp.get("start", ""),
                "endDate": exp.get("end", "") or "Present",
                "duration": self.format_duration(exp.get("start", ""), exp.get("end", "")),
                "type": "fullTime",  # optional fix
                "positionDescription": exp.get("description", "")
            })

        # Skills
        skills = [skill.get("name") for skill in input_data.get("skills", [])]

        # Sprachen
        language_skills = {}
        for lang in input_data.get("languages", []):
            language_skills[lang.get("name", "")] = lang.get("proficiency", "")

        # Komplettes Profil-Objekt
        profile_info = {
            "skills": skills,
            "firstName": first_name,
            "lastName": last_name,
            "linkedinProfile": profile_url,
            "education": education_list,
            "workExperience": work_list,
            "location": input_data.get("location", ""),
            "headline": input_data.get("headline", ""),
            "languageSkills": language_skills
        }

        # Endgültiges Mapping für CSV oder Datenbank
        final_profile = {
            "imageUrl": image_url,
            "firstName": first_name,
            "lastName": last_name,
            "name": f"{first_name} {last_name}".strip(),
            "profileLink": profile_url,
            "linkedinProfileInformation": json.dumps(profile_info, ensure_ascii=False),
            "communicationStatus": "newCandidate",
            "candidateStatus": "",
            "lastTouchpointTime": datetime.utcnow().isoformat() + "Z"
        }

        return final_profile
