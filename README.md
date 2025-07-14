# LinkedIn Karriere-Prognose KI

Ein KI-basiertes System zur Vorhersage des optimalen Zeitpunkts für den nächsten Karriereschritt basierend auf LinkedIn-Daten.

---

## Überblick

Diese Anwendung analysiert LinkedIn-Profile und prognostiziert mit Hilfe von KI-Modellen, wann ein Karriereschritt wahrscheinlich ist. Sie besteht aus einem **Backend** (Python/Flask, KI-Modelle) und einem **Frontend** (React, Material-UI).

---

## Projektstruktur (mit Erklärungen)

```
.
├── backend/                  # Backend-Server (Python, Flask, KI-Modelle)
│   ├── app.py               # Haupt-API-Server (Flask)
│   ├── config.py            # Konfiguration (z.B. Datenbank, Modellpfade)
│   ├── requirements.txt     # Python-Abhängigkeiten
│   ├── ml_pipe/             # Machine Learning Pipeline
│   │   ├── data/            # Datenverarbeitung & Feature Engineering
│   │   │   ├── featureEngineering/   # Feature-Engineering-Skripte
│   │   │   └── database/             # Datenbankanbindung (MongoDB)
│   │   ├── explainable_ai/  # SHAP & Erklärbarkeit
│   │   └── models/          # KI-Modelle (TFT, GRU, XGBoost)
│   └── api_handlers/        # API-Handler für verschiedene Endpunkte
│
├── frontend/                # React Frontend (Benutzeroberfläche)
│   ├── src/                 # React-Komponenten & Logik
│   │   ├── pages/           # Seiten (z.B. Analyse, Admin, Kandidaten)
│   │   ├── api.js           # API-URL-Konfiguration
│   │   └── ...
│   ├── public/              # Statische Dateien (HTML, Icons, ...)
│   ├── package.json         # Frontend-Abhängigkeiten
│   └── tailwind.config.js   # Tailwind CSS Konfiguration
│
├── start.sh                 # Startskript für Backend & Frontend
├── README.md                # Diese Anleitung
└── ...
```

**Wichtige Hinweise:**
- **Backend**: Alle ML-Modelle, Datenverarbeitung, API-Logik
- **Frontend**: Benutzeroberfläche, Kommunikation mit Backend
- **start.sh**: Startet beide Server automatisch

---

### Frontend
- **LinkedIn-Profil-Analyse**: Einzelne Profile analysieren und Wechselwahrscheinlichkeit berechnen
- **Batch-Upload**: Mehrere Profile per CSV hochladen und analysieren
- **Kandidatenverwaltung**: Ergebnisse durchsuchen, filtern, speichern
- **Admin-Bereich**: Nutzer anlegen, bearbeiten, löschen, Rechte verwalten
- **Erklärungen**: SHAP-basierte Feature-Importance für jede Prognose

### Backend
- **API-Server**: REST-API für alle Frontend-Funktionen
- **KI-Modelle**: Temporal Fusion Transformer (TFT), GRU, XGBoost
- **Feature Engineering**: Automatische Aufbereitung und Transformation der LinkedIn-Daten
- **Datenbank**: Speicherung von Kandidaten und Analysen in MongoDB
- **Erklärbarkeit**: SHAP-Integration für nachvollziehbare Vorhersagen

---

## Datenfluss & Verarbeitete Daten

1. **Frontend**: Nutzer gibt LinkedIn-Profil-URL oder CSV ein
2. **API-Aufruf**: Frontend sendet Daten an Backend (`/scrape-linkedin`, `/predict`, `/predict-batch`)
3. **Backend**: Holt ggf. LinkedIn-Daten, bereitet sie auf, wendet ML-Modell an
4. **Erklärung**: SHAP berechnet die wichtigsten Einflussfaktoren
5. **Antwort**: Backend sendet Prognose & Erklärungen ans Frontend
6. **Frontend**: Zeigt Ergebnisse, Visualisierungen und Erklärungen an

**Verarbeitete Daten:**
- LinkedIn-Profile (manuell oder per CSV)
- Feature-Vektoren (aus Profilen extrahiert)
- Prognoseergebnisse & SHAP-Werte
- Nutzerverwaltung (Admin-Bereich)

---

## Technologien

- **Backend:**
  - Python 3.x, Flask, Flask-CORS
  - PyTorch (TFT, GRU), XGBoost
  - SHAP (Explainable AI)
  - MongoDB (Datenbank)
- **Frontend:**
  - React, Material-UI, Tailwind CSS
  - React Router
- **DevOps:**
  - start.sh (Bash), cloudflared (optional für externe Demos)

---

## Setup & Start (Schritt für Schritt)

1. **Repository klonen**
   ```bash
   git clone [repository-url]
   cd [repository-name]
   ```
2. **Backend-Abhängigkeiten installieren**
   ```bash
   cd backend
   pip3 install -r requirements.txt
   ```
3. **Frontend-Abhängigkeiten installieren**
   ```bash
   cd ../frontend
   npm install
   ```
4. **Anwendung starten**
   ```bash
   cd ..
   chmod +x start.sh
   ./start.sh
   ```
   - Das Skript startet Backend & Frontend automatisch und öffnet das Frontend im Browser.
   - Backend: http://localhost:5100
   - Frontend: http://localhost:3000

**API-URL anpassen:**
- In `frontend/src/api.js` ggf. die URL ändern, falls Backend nicht lokal läuft.
- Für externe Demos: cloudflared nutzen und URL in `api.js` eintragen.

---

## API-Überblick (wichtigste Endpunkte)

| Methode | Pfad                | Beschreibung                        |
|---------|---------------------|-------------------------------------|
| POST    | /scrape-linkedin    | LinkedIn-Profil analysieren         |
| POST    | /predict            | Prognose für ein Profil             |
| POST    | /predict-batch      | Batch-Prognose für mehrere Profile  |
| GET     | /candidates         | Alle gespeicherten Kandidaten holen |
| POST    | /api/candidates     | Kandidaten speichern                |
| GET     | /api/users          | Nutzerliste (Admin)                 |
| POST    | /api/create-user    | Neuen Nutzer anlegen (Admin)        |
| PUT     | /api/users/:id      | Nutzer bearbeiten (Admin)           |
| DELETE  | /api/users/:id      | Nutzer löschen (Admin)              |

---

## Beispielablauf (Use-Case)

1. **Admin legt Nutzer an** (im Admin-Bereich)
2. **Nutzer lädt LinkedIn-Profil hoch** (manuell oder per CSV)
3. **Wählt Modell & startet Analyse**
4. **Erhält Prognose & Erklärungen** (z.B. Feature-Importance)
5. **Speichert interessante Kandidaten**
6. **Admin kann Nutzer verwalten, Rechte vergeben, etc.**

---

## Entwicklung & Anpassung

- **Backend:**
  - Neue Modelle in `backend/ml_pipe/models/` hinzufügen
  - Feature Engineering in `backend/ml_pipe/data/featureEngineering/`
  - API-Logik in `backend/app.py` und `backend/api_handlers/`
- **Frontend:**
  - Neue Seiten/Features in `frontend/src/pages/`
  - API-URL in `frontend/src/api.js` anpassen
  - UI/UX mit Material-UI & Tailwind erweitern

---

## Autor & E-Mail

Florian Runkel, runkel.florian@stud.uni-regensburg.de

---