FROM python:3.8-slim

# Installiere Systemabhängigkeiten
RUN apt-get update && apt-get install -y libomp-dev

# Arbeitsverzeichnis festlegen
WORKDIR /app

# Kopiere die Anforderungen in das Arbeitsverzeichnis
COPY requirements.txt .

# Installiere Python-Abhängigkeiten
RUN pip install --no-cache-dir -r requirements.txt

# Kopiere den Rest des Projekts in das Arbeitsverzeichnis
COPY . .

# Port setzen, wenn erforderlich (z.B. 5000 für Flask)
EXPOSE 5000

# Starte den Flask-Server
CMD ["python", "app.py"]
