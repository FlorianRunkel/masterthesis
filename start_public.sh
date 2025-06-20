#!/bin/bash

# Farben für die Ausgabe
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Speichere den Hauptverzeichnispfad
MAIN_DIR=$(pwd)

# Funktion zum Beenden aller Prozesse
cleanup() {
    echo -e "\n${BLUE}Beende alle Server...${NC}"
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
    fi
    echo -e "${GREEN}Alle Prozesse beendet.${NC}"
    exit 0
}

# Funktion zum Überprüfen, ob der Port verfügbar ist
wait_for_port() {
    local port=$1
    local name=$2
    local attempts=0
    local max_attempts=20 # Erhöhte Versuche für das Frontend

    echo -e "${BLUE}Warte auf ${name} auf Port ${port}...${NC}"
    while ! nc -z localhost $port >/dev/null 2>&1; do
        attempts=$((attempts + 1))
        if [ $attempts -ge $max_attempts ]; then
            echo -e "${RED}${name} konnte nicht gestartet werden (Port ${port} nicht verfügbar)${NC}"
            cleanup
            exit 1
        fi
        sleep 2
    done
    echo -e "${GREEN}${name} ist auf Port ${port} verfügbar.${NC}"
}

# Registriere cleanup für STRG+C
trap cleanup INT

echo -e "${BLUE}Starte Career Prediction Anwendung...${NC}"

# Starte Backend
echo -e "\n${GREEN}Starte Backend Server im Hintergrund...${NC}"
cd "$MAIN_DIR/backend"
/usr/local/bin/python3 app.py &
BACKEND_PID=$!

# Starte Frontend
echo -e "\n${GREEN}Starte Frontend Development Server im Hintergrund...${NC}"
cd "$MAIN_DIR/frontend"
npm start &
FRONTEND_PID=$!

# Warte bis beide Server verfügbar sind
wait_for_port 5100 "Backend"
wait_for_port 3000 "Frontend"

# Zusammenfassung der lokalen URLs
echo -e "\n${GREEN}=== LOKALE SERVER SIND BEREIT ===${NC}"
echo -e "${GREEN}Backend läuft auf: ${NC}http://localhost:5100"
echo -e "${GREEN}Frontend läuft auf: ${NC}http://localhost:3000"

# Starte Ngrok für das Frontend im Vordergrund
echo -e "\n${BLUE}Starte jetzt Ngrok für das Frontend...${NC}"
echo -e "${YELLOW}Die öffentliche URL wird unten angezeigt. Drücke STRG+C, um alles zu beenden.${NC}\n"
ngrok http http://localhost:3000

# Wenn ngrok mit STRG+C beendet wird, ruft der Trap oben die cleanup-Funktion auf.
wait 