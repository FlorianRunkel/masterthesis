#!/bin/bash

# Farben
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

MAIN_DIR=$(pwd)

cleanup() {
    echo -e "\n${BLUE}Beende alle Server...${NC}"
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null
    fi
    if [ ! -z "$NGROK_BACKEND_PID" ]; then
        kill $NGROK_BACKEND_PID 2>/dev/null
    fi
    if [ ! -z "$NGROK_FRONTEND_PID" ]; then
        kill $NGROK_FRONTEND_PID 2>/dev/null
    fi
    echo -e "${GREEN}Alle Prozesse beendet.${NC}"
    exit 0
}

trap cleanup INT

echo -e "${GREEN}Starte Backend...${NC}"
cd "$MAIN_DIR/backend"

# Virtuelle Umgebung
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt

# Backend starten
python app.py &
BACKEND_PID=$!

sleep 3

# ngrok für Backend starten
if ! command -v ngrok &> /dev/null; then
    echo -e "${RED}ngrok ist nicht installiert!${NC}"
    echo -e "${YELLOW}Installiere es von https://ngrok.com/download${NC}"
    cleanup
    exit 1
fi

echo -e "${GREEN}Starte ngrok für Backend (Port 5100)...${NC}"
ngrok http 5100 > /tmp/ngrok_backend.log 2>&1 &
NGROK_BACKEND_PID=$!
sleep 5
NGROK_BACKEND_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o '"public_url":"[^"]*"' | grep 5100 | head -1 | cut -d'"' -f4)

cd "$MAIN_DIR/frontend"

# Node Modules installieren falls nötig
if [ ! -d "node_modules" ]; then
    npm install
fi

echo -e "${GREEN}Starte Frontend...${NC}"
npm start &
FRONTEND_PID=$!

sleep 3

# ngrok für Frontend starten
echo -e "${GREEN}Starte ngrok für Frontend (Port 3000)...${NC}"
ngrok http 3000 > /tmp/ngrok_frontend.log 2>&1 &
NGROK_FRONTEND_PID=$!
sleep 5
NGROK_FRONTEND_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o '"public_url":"[^"]*"' | grep 3000 | head -1 | cut -d'"' -f4)

# Zusammenfassung
clear

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Backend läuft lokal auf:   ${NC}http://localhost:5100"
echo -e "${GREEN}Frontend läuft lokal auf:  ${NC}http://localhost:3000"
echo -e "${YELLOW}Backend öffentlich (ngrok): ${NC}$NGROK_BACKEND_URL"
echo -e "${YELLOW}Frontend öffentlich (ngrok):${NC} $NGROK_FRONTEND_URL"
echo -e "${YELLOW}Trage die Backend-ngrok-URL in dein Frontend (api.js) ein, wenn du von außen zugreifen willst!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${YELLOW}Drücke STRG+C zum Beenden aller Server${NC}"
echo ""

wait 