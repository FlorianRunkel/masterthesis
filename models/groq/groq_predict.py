import os
import requests

GROQ_API_KEY = "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"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def generate_groq_prediction(first_name, last_name, location, experiences):
    # Erstelle den Prompt-Text aus den Eingaben
    experience_text = "\n".join([f"- {exp['company']} ({exp['position']}, {exp['startDate']} - {exp['endDate']})" for exp in experiences])

    prompt = f"""
    Du bist eine KI, die Karriereempfehlungen basierend auf bisherigen Erfahrungen gibt.

    **Name:** {first_name} {last_name}  
    **Wohnort:** {location}  
    **Berufserfahrung:**  
    {experience_text}

    **Frage:** Wann wäre der optimale Zeitpunkt für den nächsten Karriereschritt und welche Position wäre empfehlenswert?

    **Antwort:**  
    Gebe ausschließlich das Datum (Monat/Jahr) und die empfohlene Position aus.  
    Falls keine Vorhersage möglich ist, antworte mit: "Aktuell keine Vorhersage möglich."
    """

    # API-Request an Groq senden
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150
    }

    response = requests.post(GROQ_API_URL, json=payload, headers=headers)

    print(response)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return "Fehler bei der Vorhersage mit Groq AI."
