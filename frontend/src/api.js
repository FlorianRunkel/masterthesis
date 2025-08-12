// Nach dem Deploy: API Proxy - CORS muss im Proxy konfiguriert werden
export const API_BASE_URL = "https://masterthesis-api-proxy.onrender.com";

// Fallback CORS-Proxy (falls der API-Proxy CORS-Probleme hat)
// export const API_BASE_URL = "https://cors-anywhere.herokuapp.com/https://masterthesis-api-proxy.onrender.com";

// Alternative CORS-Proxy
// export const API_BASE_URL = "https://api.allorigins.win/raw?url=https://masterthesis-api-proxy.onrender.com";

// Lokale Entwicklung (auskommentiert)
// export const API_BASE_URL = "http://localhost:8080";

// API Proxy (auskommentiert wegen CORS-Problemen)
// export const API_BASE_URL = "https://masterthesis-api-proxy.onrender.com";

// Cloudflare Tunnel (für lokales Testen)
// export const API_BASE_URL = "https://lot-realtors-tribune-shapes.trycloudflare.com"; 