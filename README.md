# LinkedIn Karriere-Prognose KI

Ein KI-basiertes System zur Vorhersage des optimalen Zeitpunkts für den nächsten Karriereschritt basierend auf LinkedIn-Daten.

## Projektstruktur

```
.
├── backend/                # Backend-Server
│   ├── app.py             # Flask-Anwendung
│   └── ml_pipe/           # Machine Learning Pipeline
│       ├── data/          # Datenverarbeitung
│       │   ├── featureEngineering/
│       │   │   ├── featureEngineering.py
│       │   │   └── position_level.json
│       │   └── database/
│       │       └── mongodb.py
│       └── models/        # KI-Modelle
│           ├── tft/       # Temporal Fusion Transformer
│           ├── gru/       # GRU Modell
│           └── xgboost/   # XGBoost Modell
├── frontend/              # React Frontend
│   ├── src/              # React Komponenten
│   ├── package.json      # Frontend Abhängigkeiten
│   └── public/           # Statische Dateien
└── start.sh              # Startskript für Backend und Frontend
```

## Features

- Vorhersage der Wechselwahrscheinlichkeit basierend auf LinkedIn-Profilen
- Drei verschiedene KI-Modelle zur Analyse:
  - Temporal Fusion Transformer (TFT)
  - GRU (Gated Recurrent Unit)
  - XGBoost
- Detaillierte Erklärungen der Vorhersagen
- Interaktives Dashboard mit:
  - LinkedIn-Profil-Analyse
  - Batch-Upload für mehrere Profile
  - Kandidatenübersicht
- MongoDB-Integration für Kandidatenverwaltung

## Installation & Start

### Voraussetzungen

- Python 3.x
- Node.js und npm
- MongoDB (lokal oder Remote)

### Schnellstart

1. Repository klonen:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Backend-Abhängigkeiten installieren:
```bash
cd backend
pip3 install flask flask-cors pandas numpy linkedin-api
```

3. Frontend-Abhängigkeiten installieren:
```bash
cd ../frontend
npm install
```

4. Anwendung starten:
```bash
cd ..
chmod +x start.sh
./start.sh
```

Die Anwendung ist dann unter folgenden URLs erreichbar:
- Frontend: http://localhost:3000
- Backend: http://localhost:5100

## Verwendung

### LinkedIn-Analyse

1. Navigieren Sie zur "LinkedIn Analyse"-Seite
2. Fügen Sie einen LinkedIn-Profil-Link ein
3. Wählen Sie das gewünschte KI-Modell
4. Klicken Sie auf "Analysieren"

### Batch-Upload

1. Navigieren Sie zur "Batch Upload"-Seite
2. Laden Sie eine CSV-Datei mit LinkedIn-Profilen hoch
3. Starten Sie die Batch-Analyse
4. Speichern Sie interessante Kandidaten

### Kandidatenverwaltung

1. Navigieren Sie zur "Kandidaten"-Seite
2. Sehen Sie alle gespeicherten Kandidaten
3. Filtern und sortieren Sie nach verschiedenen Kriterien

## API-Endpunkte

- `POST /scrape-linkedin`: LinkedIn-Profil analysieren
- `POST /predict`: Vorhersage für ein Profil erstellen
- `POST /predict-batch`: Batch-Vorhersage für mehrere Profile
- `GET /candidates`: Alle gespeicherten Kandidaten abrufen
- `POST /api/candidates`: Neue Kandidaten speichern

## Entwicklung

### Backend-Entwicklung

```bash
cd backend
python3 app.py
```

### Frontend-Entwicklung

```bash
cd frontend
npm start
```

## Technologien

- **Backend**:
  - Python 3.x
- Flask
  - PyTorch (TFT, GRU)
  - XGBoost
  - MongoDB

- **Frontend**:
  - React
  - Material-UI
  - React Router
  - Tailwind CSS

## Autor

Florian Runkel


