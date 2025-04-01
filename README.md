# LinkedIn Karriere-Prognose KI

Ein KI-basiertes System zur Vorhersage des optimalen Zeitpunkts für den nächsten Karriereschritt basierend auf LinkedIn-Daten.

## Projektstruktur

```
.
├── app.py                 # Flask-Anwendung
├── ml_pipe/              # Machine Learning Pipeline
│   ├── data/            # Datenverarbeitung
│   │   ├── dummy_data.py
│   │   ├── datamodule.py
│   │   └── data_processor.py
│   ├── models/          # KI-Modelle
│   │   └── model.py
│   └── predict.py       # Vorhersage-Logik
├── dashboard/           # Frontend
│   ├── static/
│   │   ├── css/
│   │   └── js/
│   └── templates/
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Features

- Vorhersage des nächsten Karriereschritts
- Konfidenz-Score für Vorhersagen
- Personalisierte Empfehlungen
- Interaktives Dashboard
- Dummy-Daten für Tests und Entwicklung

## Installation

### Option 1: Lokale Installation

1. Repository klonen:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Virtuelle Umgebung erstellen und aktivieren:
```bash
python -m venv venv
source venv/bin/activate  # Unter Windows: venv\Scripts\activate
```

3. Abhängigkeiten installieren:
```bash
pip install -r requirements.txt
```

4. Anwendung starten:
```bash
python app.py
```

### Option 2: Docker Installation

1. Docker und Docker Compose installieren

2. Repository klonen:
```bash
git clone [repository-url]
cd [repository-name]
```

3. Container starten:
```bash
docker-compose up --build
```

## Verwendung

1. Öffnen Sie die Anwendung im Browser: `http://localhost:5001`

2. Geben Sie Ihre persönlichen Informationen ein:
   - Name
   - Standort
   - Berufserfahrung
   - Ausbildung

3. Wählen Sie das KI-Modell aus:
   - LLM

4. Klicken Sie auf "Prognose erstellen"

5. Die Vorhersage wird mit Konfidenz-Score und Empfehlungen angezeigt

## API-Endpunkte

- `GET /api/profiles`: Alle Profile abrufen
- `GET /api/experiences/<profile_id>`: Berufserfahrung eines Profils abrufen
- `GET /api/education/<profile_id>`: Ausbildung eines Profils abrufen
- `POST /predict`: Neue Vorhersage generieren

## Entwicklung

### Dummy-Daten generieren

```bash
python ml_pipe/data/dummy_data.py
```

### Tests ausführen

```bash
python -m pytest tests/
```

## Technologien

- Python 3.9
- Flask
- PyTorch
- SQLite
- Docker
- HTML/CSS/JavaScript

## Autor

Florian Runkel


