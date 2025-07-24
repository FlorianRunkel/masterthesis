from datetime import datetime
from dateutil.relativedelta import relativedelta

def smart_parse_date(date_str):
    """
    Versucht, das Datum als DD/MM/YYYY und MM/DD/YYYY zu parsen.
    Gibt das plausibelste Datum zurück (z.B. nicht in der Zukunft).
    """
    if not date_str or date_str.lower() == "present":
        return None

    now = datetime.now()
    candidates = []
    for fmt in ("%d/%m/%Y", "%m/%d/%Y"):
        try:
            dt = datetime.strptime(date_str, fmt)
            # Plausibilitätscheck: nicht in der Zukunft, nicht zu alt
            if dt <= now and dt.year > 1950:
                candidates.append(dt)
        except Exception:
            continue
    # Fallback: andere Formate probieren
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%Y", "%Y"):
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt <= now and dt.year > 1950:
                candidates.append(dt)
        except Exception:
            continue
    # Wenn mehrere Kandidaten: nimm das jüngste Datum
    if candidates:
        return max(candidates)
    return None

class CareerRules:
    @staticmethod
    def is_last_position_too_new(career_history, min_months=6):
        """
        Prüft, ob die letzte (aktuelle) Position jünger als min_months ist.
        Gibt (True/False, Monate seit Start) zurück.
        """
        if not career_history or not isinstance(career_history, list) or len(career_history) == 0:
            return False, None
        last_pos = career_history[0]
        print(f"Last position: {last_pos}")
        
        # Unterstütze verschiedene Feldnamen (start_date, startDate, etc.)
        start_date = last_pos.get('start_date') or last_pos.get('startDate')
        end_date = last_pos.get('end_date') or last_pos.get('endDate', 'Present')
        
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