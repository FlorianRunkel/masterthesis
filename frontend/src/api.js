// Axios-Konfiguration für erhöhte Timeouts
import axios from 'axios';

// Nach dem Deploy: API Proxy - CORS muss im Proxy konfiguriert werden
export const API_BASE_URL = "https://masterthesis-api-proxy.onrender.com";

// Intelligente Timeout-Konfiguration
const TIMEOUTS = {
  // Basis-Timeouts (für normale Requests)
  DEFAULT: 30000,        // 30 Sekunden
  MEDIUM: 60000,         // 1 Minute
  
  // Erhöhte Timeouts (für rechenintensive Operationen)
  LINKEDIN_SCRAPING: 180000,  // 3 Minuten
  ML_PREDICTION: 300000,      // 5 Minuten
  BATCH_PROCESSING: 600000,   // 10 Minuten
  
  // Adaptive Timeouts (passen sich an)
  ADAPTIVE: {
    START: 30000,        // Start mit 30 Sekunden
    MAX: 300000,         // Maximum 5 Minuten
    MULTIPLIER: 1.5      // Bei Timeout: 1.5x erhöhen
  }
};

// Globale Axios-Konfiguration
axios.defaults.timeout = TIMEOUTS.DEFAULT;
axios.defaults.timeoutErrorMessage = 'Request timed out. Please try again.';

// Intelligenter Timeout-Interceptor
axios.interceptors.response.use(
  (response) => response,
  async (error) => {
    if (error.code === 'ECONNABORTED') {
      console.error('Request timeout:', error.message);
      
      // Adaptive Timeout-Erhöhung
      const currentTimeout = error.config.timeout || TIMEOUTS.DEFAULT;
      const newTimeout = Math.min(
        currentTimeout * TIMEOUTS.ADAPTIVE.MULTIPLIER,
        TIMEOUTS.ADAPTIVE.MAX
      );
      
      console.log(`Increasing timeout from ${currentTimeout}ms to ${newTimeout}ms`);
      
      // Request mit neuem Timeout wiederholen
      error.config.timeout = newTimeout;
      return axios.request(error.config);
    }
    return Promise.reject(error);
  }
);

// Export der Timeouts für andere Komponenten
export { TIMEOUTS };

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