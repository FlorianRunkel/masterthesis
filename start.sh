#!/bin/bash

# Farben für die Ausgabe
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Speichere den Hauptverzeichnispfad
MAIN_DIR=$(pwd)

# Funktion zum Beenden aller Prozesse
cleanup() {
    echo -e "\n${BLUE}Beende alle Server...${NC}"
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID
    fi
    exit 0
}

# Funktion zum Überprüfen, ob der Port verfügbar ist
wait_for_port() {
    local port=$1
    local attempts=0
    local max_attempts=10

    while ! nc -z localhost $port >/dev/null 2>&1; do
        attempts=$((attempts + 1))
        if [ $attempts -ge $max_attempts ]; then
            echo -e "${RED}Backend-Server konnte nicht gestartet werden (Port $port nicht verfügbar)${NC}"
            cleanup
            exit 1
        fi
        echo -e "${BLUE}Warte auf Backend-Server (Versuch $attempts von $max_attempts)...${NC}"
        sleep 2
    done
}

# Registriere cleanup für STRG+C
trap cleanup INT

echo -e "${BLUE}Starte Career Prediction Anwendung...${NC}"

# Starte Backend
echo -e "${GREEN}Starte Backend Server...${NC}"
cd "$MAIN_DIR/backend"
/usr/local/bin/python3 app.py &
BACKEND_PID=$!

# Warte bis Backend verfügbar ist
wait_for_port 5100

# Starte Frontend
echo -e "${GREEN}Starte Frontend Development Server...${NC}"
cd "$MAIN_DIR/frontend"
npm start &
FRONTEND_PID=$!

# Öffne Frontend im Browser
if which xdg-open > /dev/null; then
  xdg-open http://localhost:3000
elif which open > /dev/null; then
  open http://localhost:3000
elif which start > /dev/null; then
  start http://localhost:3000
else
  echo -e "${RED}Konnte Browser nicht automatisch öffnen. Bitte öffne http://localhost:3000 manuell.${NC}"
fi

echo -e "${BLUE}Beide Server wurden gestartet${NC}"
echo -e "${GREEN}Backend läuft auf: ${NC}http://localhost:5100"
echo -e "${GREEN}Frontend läuft auf: ${NC}http://localhost:3000"
echo -e "${BLUE}Drücke STRG+C zum Beenden beider Server${NC}"

# Warte auf Beendigung
wait 