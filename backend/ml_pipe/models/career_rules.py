import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
import logging
import random

logger = logging.getLogger(__name__)

MONTHS = {
    'januar': 1, 'jan': 1, 'january': 1,
    'februar': 2, 'feb': 2, 'february': 2,
    'märz': 3, 'maerz': 3, 'marz': 3, 'mar': 3, 'march': 3,
    'april': 4, 'apr': 4,
    'mai': 5, 'may': 5,
    'juni': 6, 'jun': 6, 'june': 6,
    'juli': 7, 'jul': 7, 'july': 7,
    'august': 8, 'aug': 8,
    'september': 9, 'sep': 9, 'sept': 9,
    'oktober': 10, 'okt': 10, 'october': 10, 'oct': 10,
    'november': 11, 'nov': 11,
    'dezember': 12, 'dez': 12, 'december': 12, 'dec': 12
}

'''
Replace month names with numbers
'''
def normalize_month_names(date_str):
    try:
        for name, num in MONTHS.items():
            pattern = re.compile(rf'\b{name}\b', re.IGNORECASE)
            date_str = pattern.sub(str(num), date_str)
            return date_str
    except Exception:
        logger.error(f"Error normalizing month names: {date_str}")
        return date_str

'''
Parse a date string in various formats
'''
def parse_flexible_date(date_str):
    try:
        if not date_str or str(date_str).strip().lower() in ["present", "heute", "now", "aktuell"]:
            return None
        date_str = date_str.strip().lower()
        date_str = normalize_month_names(date_str)
        date_str = re.sub(r"[\.\-\s]", "/", date_str)
        now = datetime.now()
        future_limit = now.replace(year=now.year + 2)
        patterns = [
            "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
            "%d/%m/%y", "%m/%d/%y", "%y/%m/%d",
            "%Y/%m", "%m/%Y", "%Y",
            "%d/%m", "%m/%d",
        ]
        for fmt in patterns:
            try:
                dt = datetime.strptime(date_str, fmt)
                if dt.year < 1950 or dt > future_limit:
                    continue
                return dt
            except Exception:
                continue
        year_match = re.match(r"^(\d{4})$", date_str)
        if year_match:
            return datetime(int(year_match.group(1)), 1, 1)
        return None
    except Exception:
        logger.error(f"Error parsing date: {date_str}")
        return None

'''
Compare two companies
'''
def companies_are_equal(pos1, pos2):
    try:
        c1 = (pos1.get('company') or '').strip().lower()
        c2 = (pos2.get('company') or '').strip().lower()
        return bool(c1 and c2 and c1 == c2)
    except Exception:
        logger.error(f"Error comparing companies: {pos1} and {pos2}")
        return False

'''
Calculate the months between two dates
'''
def months_between_dates(start_dt, end_dt):
    try:
        diff = relativedelta(end_dt, start_dt)
        return diff.years * 12 + diff.months + diff.days / 30.44
    except Exception:
        logger.error(f"Error calculating months between dates: {start_dt} and {end_dt}")
        return 0

class CareerRules:

    '''
    Check if the last position is too new
    '''
    @staticmethod
    def is_last_position_too_new(career_history, min_months=6, model="gru"):
        try:
            if not career_history or not isinstance(career_history, list) or len(career_history) == 0:
                return False, None
            last_pos = career_history[0]

            start_date = last_pos.get('start_date') or last_pos.get('startDate')
            end_date = last_pos.get('end_date') or last_pos.get('endDate', 'Present')

            if len(career_history) > 1 and companies_are_equal(last_pos, career_history[1]):
                return False, None
            if not start_date:
                return False, None
            start_dt = parse_flexible_date(start_date)
            if not start_dt:
                return False, None
            now = datetime.now()
            if start_dt > now:
                return False, None
            if end_date == 'Present':
                end_dt = now
            else:
                end_dt = parse_flexible_date(end_date)
                if not end_dt:
                    end_dt = now
            months = months_between_dates(start_dt, end_dt)
            # Regel nur für direkt nach der Probezeit (6-9 Monate)
            if 6 <= months < 9:
                info = {
                    "confidence": [0] if model == "xgboost" else [random.randint(370, 500)],  # oder [0] oder [400] je nach Modell
                    "recommendations": [
                        "The current position is too new for a change.",
                        f"Months in current position: {months:.1f}"
                    ],
                    "status": "Very unlikely",
                    "shap_explanations": [{
                        "feature": "duration current position",
                        "impact_percentage": 100.0,
                        "method": "SHAP",
                        "description": "The current position is too new for a change."
                    }],
                    "shap_summary": "",
                    "lime_explanations": [{
                        "feature": "duration current position",
                        "impact_percentage": 100.0,
                        "method": "LIME",
                        "description": "The current position is too new for a change."
                    }],
                    "lime_summary": "",
                    "llm_explanation": f"Candidate is too new in the current position. Months in current position: {months:.1f}"
                }
                return True, info
            return False, None
        except Exception:
            logger.error(f"Error checking if last position is too new: {career_history}")
            return False, None

    '''
    Check if the last position is a C-Level or Founder position
    '''
    @staticmethod
    def is_c_level_or_founder(career_history, model="gru"):
        try:
            if not career_history or not isinstance(career_history, list) or len(career_history) == 0:
                return False, None
            last_pos = career_history[0]
            title = (last_pos.get('position') or last_pos.get('title') or '').lower()
            keywords = [
                "founder", "co-founder", "cofounder", "owner", "partner",
                "ceo", "cfo", "coo", "cto", "cmo", "cio", "cso", "cdo", "cro",
                "chief executive officer", "chief financial officer", "chief operations officer",
                "chief technology officer", "chief marketing officer", "chief information officer",
                "chief strategy officer", "chief revenue officer", "chief digital officer",
                "president", "vice president", "vp", "chairman", "board member", "managing director",
                "geschäftsführer", "geschaeftsführer", "inhaber", "mitgründer", "mitgruender", "mit-inhaber",
                "vorstand", "direktor", "geschäftsleitung", "unternehmensleitung"
            ]
            for kw in keywords:
                if kw in title:
                    info = {
                        "confidence": [0] if model == "xgboost" else [random.randint(1800, 2190)],
                        "recommendations": [
                            "C-Level/Founder-Position: Ein Wechsel ist sehr unwahrscheinlich."
                        ],
                        "status": "Very unlikely",
                        "shap_explanations": [{
                            "feature": "current position title",
                            "impact_percentage": 100.0,
                            "method": "SHAP",
                        }],
                        "shap_summary": "",
                        "lime_explanations": [{
                            "feature": "current position title",
                            "impact_percentage": 100.0,
                            "method": "LIME",
                        }],
                        "lime_summary": "",
                        "llm_explanation": ""
                    }
                    return True, info
            return False, None
        except Exception:
            logger.error(f"Error checking C-Level/Founder rule: {career_history}")
            return False, None

    '''
    Check if the last position is a sabbatical or gap year
    '''
    @staticmethod
    def is_on_sabbatical_or_gap_year(career_history, model="gru"):
        try:
            if not career_history or not isinstance(career_history, list) or len(career_history) == 0:
                return False, None
            last_pos = career_history[0]
            title = (last_pos.get('position') or last_pos.get('title') or '').lower()
            keywords = [
                'sabbatical', 'gap year', 'auszeit', 'pause', 'career break',
                'sabbatjahr', 'sabbatical leave', 'berufliche pause', 'sabbaticaljahr'
            ]
            print("DEBUG SABBATICAL TITLE:", title)
            for kw in keywords:
                if kw in title:
                    info = {
                        "confidence": [0] if model == "xgboost" else [random.randint(200, 400)],
                        "recommendations": [
                            "Candidate is currently in a sabbatical or gap year."
                        ],
                        "status": "Special case",
                        "shap_explanations": [{
                            "feature": "current position title",
                            "impact_percentage": 100.0,
                            "method": "SHAP",
                        }],
                        "shap_summary": "",
                        "lime_explanations": [{
                            "feature": "current position title",
                            "impact_percentage": 100.0,
                            "method": "LIME",
                        }],
                        "lime_summary": "",
                        "llm_explanation": ""
                    }
                    return True, info
            return False, None
        except Exception:
            logger.error(f"Error checking sabbatical/gap year rule: {career_history}")
            return False, None

    '''
    Check if the candidate has been in the current position for more than 1 year AND has multiple positions at the same company (career progression indicator)
    '''
    @staticmethod
    def is_long_term_current_position(career_history, min_years=1, model="gru"):
        try:
            if not career_history or not isinstance(career_history, list) or len(career_history) == 0:
                return False, None

            current_position = career_history[0]
            current_company = (current_position.get('company') or '').strip()
            current_title = (current_position.get('position') or '').strip()

            if not current_company or not current_title:
                return False, None

            positions_at_current_company = []
            for position in career_history:
                company = (position.get('company') or '').strip()
                if company == current_company:
                    positions_at_current_company.append(position)

            if len(positions_at_current_company) < 2:
                return False, None

            total_months_at_company = 0

            for position in positions_at_current_company:
                start_date = position.get('start_date') or position.get('startDate')
                end_date = position.get('end_date') or position.get('endDate', 'Present')

                if not start_date:
                    continue

                start_dt = parse_flexible_date(start_date)
                if not start_dt:
                    continue

                if end_date == 'Present':
                    end_dt = datetime.now()
                else:
                    end_dt = parse_flexible_date(end_date)
                    if not end_dt:
                        end_dt = datetime.now()

                months = months_between_dates(start_dt, end_dt)
                total_months_at_company += months

            total_years_at_company = total_months_at_company / 12

            if total_years_at_company >= min_years:
                confidence_value = 0 if model == "xgboost" else random.randint(400, 600)

                info = {
                    "confidence": [confidence_value],
                    "recommendations": [],
                    "status": "Unlikely",
                    "shap_explanations": [{
                        "feature": "company loyalty",
                        "impact_percentage": 100.0,
                        "method": "SHAP",
                    }],
                    "shap_summary": "",
                    "lime_explanations": [{
                        "feature": "company loyalty",
                        "impact_percentage": 100.0,
                        "method": "LIME",
                    }],
                    "lime_summary": "",
                    "llm_explanation": ""
                }
                return True, info
            return False, None
        except Exception:
            logger.error(f"Error checking long-term current position: {career_history}")
            return False, None

    '''
    Check all rules
    '''
    @staticmethod
    def check_all_rules(career_history, **kwargs):
        rules = [
            lambda ch, **kw: CareerRules.is_on_sabbatical_or_gap_year(ch, model=kw.get('model', 'gru')),
            lambda ch, **kw: CareerRules.is_c_level_or_founder(ch, model=kw.get('model', 'gru')),
            lambda ch, **kw: CareerRules.is_long_term_current_position(ch, min_years=kw.get('min_years', 1), model=kw.get('model', 'gru')),
            lambda ch, **kw: CareerRules.is_last_position_too_new(ch, min_months=kw.get('min_months', 6), model=kw.get('model', 'gru')),
            # add rules
        ]
        for rule in rules:
            result, info = rule(career_history, **kwargs)
            if result:
                return True, info
        return False, None