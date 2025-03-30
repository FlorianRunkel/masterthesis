# masterthesis

Diese Anwendung ist Teil der Masterarbeit "Predictive Talent Analytics: Leveraging AI to Anticipate Career Shifts". Ziel der Forschung ist es, mit Hilfe künstlicher Intelligenz den optimalen Zeitpunkt für den nächsten Karriereschritt vorherzusagen. Dazu wird ein innovatives KI-Modell entwickelt, das Karrieredaten analysiert und präzise Prognosen ermöglicht.

## Ausgangslage der Masterarbeit

Die Herausforderung, die Wechselbereitschaft von Talenten frühzeitig zu erkennen, ist ein zentraler Bestandteil der Forschung. Im Rahmen dieser Arbeit werden folgende Forschungsfragen untersucht:

1. **Datengrundlage:** Welche Daten basierend auf LinkedIn-Profilen sind erforderlich, um KI-Modelle zu entwickeln?
2. **Analyse der Modellperformance:** Welche Methoden eignen sich zur Verbesserung der Performance?
3. **Active-Sourcing:** Wie können die Ergebnisse auf das Active-Sourcing angewendet werden?

Diese Admin-App dient als Testumgebung für verschiedene Modelle, darunter Groq und ein Custom LLM.

## Installation & Setup

### Voraussetzungen
- Docker
- Python 3.x
- Flask
- Docker Compose 

### Installation
1. Repository klonen:
   ```bash
   git clone <repository-url>
   cd admin-app
   ```
2. Virtuelle Umgebung erstellen (optional):
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate  # Windows
   ```
3. Abhängigkeiten installieren:
   ```bash
   pip install -r requirements.txt
   ```
4. Anwendung starten:
   ```bash
   python app.py
   ```

## Nutzung

### Endpunkte
- `/` - Index-Seite für die Admin-App
- `/about` - About-Seite
- `/models` (GET) - Gibt eine Liste der verfügbaren Modelle zurück
- `/run_model` (POST) - Führt ein Modell mit einem gegebenen Input aus

### Beispiel für eine API-Anfrage
```bash
curl -X POST http://127.0.0.1:5000/run_model \
     -H "Content-Type: application/json" \
     -d '{"model": "Groq", "input": "Karriereprognose"}'
```

## Docker
### Anwendung mit Docker starten
1. Docker-Container bauen:
   ```bash
   docker build -t admin-app .
   ```
2. Container starten:
   ```bash
   docker run -p 5000:5000 admin-app
   ```

### Docker Compose
Falls eine `docker-compose.yml` vorhanden ist, kann die Anwendung mit folgendem Befehl gestartet werden:
```bash
docker-compose up
```


