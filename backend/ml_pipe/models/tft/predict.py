import torch
import json
from datetime import datetime
import sys
import os

sys.path.insert(0, '/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/')
from backend.ml_pipe.models.tft.model import TFTModel

# Zeiträume wie in classify_change
ZEITRAUM_LABELS = {
    0: "0-6 Monate",
    1: "7-12 Monate",
    2: "13-24 Monate",
    3: "über 24 Monate"
}

def parse_date(date_str):
    if date_str == "Present":
        return datetime.now()
    try:
        return datetime.strptime(date_str, "%d/%m/%Y")
    except Exception:
        try:
            return datetime.strptime(date_str, "%m/%Y")
        except Exception:
            try:
                return datetime.strptime(date_str, "%Y")
            except Exception:
                try:
                    return datetime.strptime(date_str, "%Y-%m-%d")  # ISO-Format!
                except Exception:
                    return None

def get_latest_model_path(saved_models_dir):
    # Finde die neueste .pt-Datei im Verzeichnis
    pt_files = [f for f in os.listdir(saved_models_dir) if f.endswith('.pt')]
    if not pt_files:
        raise FileNotFoundError("Kein Modell im Verzeichnis gefunden.")
    pt_files = [os.path.join(saved_models_dir, f) for f in pt_files]
    latest_file = max(pt_files, key=os.path.getmtime)
    return latest_file

def predict(profile, model_path=None, mapping_path=None, with_llm_explanation=False):
    try:
        # Falls String, in Dict umwandeln
        if isinstance(profile, str):
            profile = json.loads(profile)
        work_experience = profile["workExperience"]

        # Nach Startdatum sortieren (neueste zuerst)
        work_experience_sorted = sorted(
            work_experience,
            key=lambda x: parse_date(x["startDate"]) or datetime(1900, 1, 1),
            reverse=True
        )

        # Aktuellste Position extrahieren
        last_position = work_experience_sorted[0]
        positionsname = last_position["position"]
        start = parse_date(last_position["startDate"])
        end = parse_date(last_position["endDate"])
        if end is None or end == "Present":
            end = datetime.now()
        wechselzeitraum = (end.year - start.year) * 12 + (end.month - start.month)

        # Mapping laden
        if mapping_path is None:
            mapping_path = "/Users/florianrunkel/Documents/02_Uni/04_Masterarbeit/masterthesis/backend/ml_pipe/data/dataModule/tft/position_to_idx.json"
        with open(mapping_path, "r") as f:
            position_to_idx = json.load(f)

        if positionsname not in position_to_idx:
            pos_idx = position_to_idx.get("UNK", 0)
        else:
            pos_idx = position_to_idx[positionsname]

        # Eingabevektor für das Modell bauen
        x_seq = torch.tensor([[pos_idx, wechselzeitraum]], dtype=torch.float32)

        # Modellpfad automatisch bestimmen, falls nicht angegeben
        if model_path is None:
            saved_models_dir = os.path.join(os.path.dirname(__file__), "saved_models")
            model_path = get_latest_model_path(saved_models_dir)

        # Modell laden
        model = TFTModel(sequence_features=2, hidden_size=64, dropout=0.1)
        model.load_state_dict(torch.load(model_path)["model_state_dict"])
        model.eval()

        # Vorhersage
        with torch.no_grad():
            pred = model(x_seq)
            pred_class = torch.argmax(pred, dim=1).item()  # 0-3
            softmax_probs = torch.softmax(pred, dim=1).numpy()[0]
            pred_value = float(softmax_probs[pred_class])

        zeitraum = ZEITRAUM_LABELS.get(pred_class, "unbekannt")

        return {
            "confidence": softmax_probs.tolist(),
            "recommendations": zeitraum,
            "status": "success"
        }
    except FileNotFoundError as e:
        print(f"Fehler bei der Vorhersage: {str(e)}")
        return {
            "confidences": [0.0],
            "zeitraum": "unbekannt",
            "status": "error",
            "error": str(e)
        }

# Beispielaufruf:
'''
if __name__ == "__main__":
    # Hier den JSON-String aus deiner CSV einfügen
    profile_str = r{"skills":["Multitasking","Kundenservice","Interpersonelle Fähigkeiten","Kaltakquise","Hubspot CRM","Customer-Relationship-Management (CRM)"],"firstName":"Darya","lastName":"Chernuska","profilePicture":"https://media.licdn.com/dms/image/v2/D4E03AQE0yuZ6cg8f4A/profile-displayphoto-shrink_100_100/profile-displayphoto-shrink_100_100/0/1670856025914?e=1749686400&v=beta&t=jI1mkiVnkD7teWPncsg8QtKAwZKB-az53_4ny7C7XvI","linkedinProfile":"https://www.linkedin.com/in/daryachernuska","education":[{"duration":"01/01/2017 - 01/01/2022","institution":"Ludwig-Maximilians-Universität München","endDate":"01/01/2022","degree":"","startDate":"01/01/2017"}],"providerId":"ACoAAD0rz_IBI0XfqqBDUscwHoFwuOqJa_c5T2I","workExperience":[{"duration":"01/03/2023 - Present","endDate":"Present","companyInformation":{"employee_count":515,"activities":["Telefonie","Internet","Vernetzung","Rechenzentrum","Glasfaser","Highspeed-Internet","Business-Internet","SIP-Trunk","Cloud-Lösungen","Connect-Cloud","Connect-LAN","Premium IP","Internet + Telefonie","Lösungen für Geschäftskunden"],"name":"M-net Telekommunikations GmbH","description":"Als regionaler Telekommunikationsanbieter versorgt M-net große Teile Bayerns, den Großraum Ulm sowie weite Teile des hessischen Landkreises Main-Kinzig mit zukunftssicherer Kommunikationstechnologie.","industry":["Telecommunications"]},"description":"","company":"M-net Telekommunikations GmbH","location":"München, Bayern, Deutschland · Hybrid","position":"Disponentin","startDate":"01/03/2023"},{"duration":"01/08/2022 - 01/12/2022","endDate":"01/12/2022","companyInformation":{"employee_count":2048,"activities":["HR Software","HR Management","Recruitung","Employee Management","Applicant Tracking System","Employee Selfservice","Time-Off Management","Cloud Software","Onboarding and Offboarding","HR Reporting","Performance Management","Payroll","HR","HR Tech","Human Resources"],"name":"Personio","description":"Personio's Intelligent HR Platform helps small and medium-sized organizations unlock the power of people by making complicated, time-consuming tasks simple and efficient.","industry":["Software Development"]},"description":"","company":"Personio","location":"München, Bayern, Deutschland","position":"Sales Development Representative","startDate":"01/08/2022"},{"duration":"01/11/2017 - 01/07/2022","endDate":"01/07/2022","companyInformation":{"employee_count":662,"activities":["Scandinavian design","Furniture","Design","Product design","Retail","Web","Steelcase partner","Wholesale","B2B","Contract sales","Online","Digital","Creativity"],"name":"BOLIA","description":"Our collection is inspired by the vivid Scandinavian nature","industry":["Retail Furniture and Home Furnishings"]},"description":"","company":"Bolia.com","location":"München, Bayern, Deutschland","position":"Sales Consultant","startDate":"01/11/2017"},{"duration":"01/10/2015 - 01/11/2017","endDate":"01/11/2017","companyInformation":{},"description":"","company":"Pepperminds","location":"München, Bayern, Deutschland","position":"Senior Team Lead","startDate":"01/10/2015"}],"location":"Munich, Bavaria, Germany","certifications":[],"headline":"-","languageSkills":{}}
    # model_path=None sorgt dafür, dass das neueste Modell automatisch gewählt wird
    result = predict(profile_str, model_path=None)
    print(json.dumps(result, indent=2, ensure_ascii=False))
'''