// Axios-Konfiguration für erhöhte Timeouts und Retry-Logik
import axios from 'axios';

// Nach dem Deploy: API Proxy - CORS muss im Proxy konfiguriert werden
export const API_BASE_URL = "https://masterthesis-api-proxy.onrender.com";

// Globale Axios-Konfiguration mit sehr hohen Timeouts für ML-Modelle
axios.defaults.timeout = 600000; // 10 Minuten
axios.defaults.timeoutErrorMessage = 'Request timed out. Please try again.';

// Retry-Logik für 504-Fehler
const retryRequest = async (config, retries = 3, delay = 2000) => {
  try {
    return await axios(config);
  } catch (error) {
    if (retries > 0 && (error.response?.status === 504 || error.code === 'ECONNABORTED')) {
      console.log(`Retry attempt ${4 - retries} for ${config.url}`);
      await new Promise(resolve => setTimeout(resolve, delay));
      return retryRequest(config, retries - 1, delay * 1.5);
    }
    throw error;
  }
};

// Interceptor für besseres Error-Handling
axios.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.code === 'ECONNABORTED') {
      console.error('Request timeout:', error.message);
      return Promise.reject(new Error('Request timed out. Please try again.'));
    }
    
    if (error.response?.status === 504) {
      console.error('Gateway timeout (504):', error.message);
      return Promise.reject(new Error('Server timeout - die Berechnung dauert zu lange. Versuchen Sie es mit weniger Daten oder warten Sie und versuchen Sie es erneut.'));
    }
    
    return Promise.reject(error);
  }
);

// Spezielle Funktionen für verschiedene API-Calls mit angepassten Timeouts
export const apiCall = {
  // Für schnelle Calls (Login, etc.)
  quick: (config) => axios({ ...config, timeout: 30000 }),
  
  // Für normale Calls
  normal: (config) => axios({ ...config, timeout: 120000 }),
  
  // Für ML-Predictions (sehr hohe Timeouts)
  ml: (config) => retryRequest({ ...config, timeout: 600000 }),
  
  // Für Batch-Uploads (höchste Timeouts)
  batch: (config) => retryRequest({ ...config, timeout: 900000 }), // 15 Minuten
};

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