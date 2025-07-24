import re
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Mapping für Monatsnamen (deutsch/englisch, kurz/lang)
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

def normalize_month_names(date_str):
    # Ersetze Monatsnamen durch Zahl
    for name, num in MONTHS.items():
        pattern = re.compile(rf'\b{name}\b', re.IGNORECASE)
        date_str = pattern.sub(str(num), date_str)
    return date_str

def smart_parse_date(date_str):
    if not date_str or str(date_str).strip().lower() in ["present", "heute", "now", "aktuell"]:
        return None

    date_str = date_str.strip().lower()
    date_str = normalize_month_names(date_str)
    # Vereinheitliche Trennzeichen
    date_str = re.sub(r"[\.\-\s]", "/", date_str)

    now = datetime.now()
    future_limit = now.replace(year=now.year + 2)  # max. 2 Jahre in die Zukunft

    # Alle Formate, die wir probieren wollen
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

    # Fallback: Versuche nur Jahr zu extrahieren
    year_match = re.match(r"^(\d{4})$", date_str)
    if year_match:
        return datetime(int(year_match.group(1)), 1, 1)

    return None

class CareerRules:
    @staticmethod
    def is_last_position_too_new(career_history, min_months=6):
        """
        Prüft, ob die letzte (aktuelle) Position jünger als min_months ist.
        Gibt (True/False, Monate seit Start) zurück.
        Ausnahme: Wenn die aktuelle Firma gleich der vorherigen Firma ist, wird immer False zurückgegeben.
        """
        if not career_history or not isinstance(career_history, list) or len(career_history) == 0:
            return False, None
        last_pos = career_history[0]
        print(f"Last position: {last_pos}")
        
        # Unterstütze verschiedene Feldnamen (start_date, startDate, etc.)
        start_date = last_pos.get('start_date') or last_pos.get('startDate')
        end_date = last_pos.get('end_date') or last_pos.get('endDate', 'Present')
        
        # Ausnahme: Wenn aktuelle und vorherige Firma gleich sind, immer False
        if len(career_history) > 1:
            prev_pos = career_history[1]
            curr_company = (last_pos.get('company') or '').strip().lower()
            prev_company = (prev_pos.get('company') or '').strip().lower()
            if curr_company and prev_company and curr_company == prev_company:
                return False, None
        
        if not start_date:
            return False, None
        # Robustere Datumserkennung
        start_dt = smart_parse_date(start_date)
        if not start_dt:
            return False, None
        # Sonderfall: Startdatum in der Zukunft
        now = datetime.now()
        if start_dt > now:
            return False, None
        if end_date == 'Present':
            end_dt = now
        else:
            end_dt = smart_parse_date(end_date)
            if not end_dt:
                end_dt = now
        diff = relativedelta(end_dt, start_dt)
        months = diff.years * 12 + diff.months + diff.days / 30.44
        return months < min_months, months 