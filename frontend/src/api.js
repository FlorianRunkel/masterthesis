import axios from 'axios';

/*
API Base URL
*/
export const API_BASE_URL = "https://masterthesis-api-proxy.onrender.com";

/*
Global Axios configuration with very high timeouts for ML models
*/
axios.defaults.timeout = 600000; // 10 Minuten
axios.defaults.timeoutErrorMessage = 'Request timed out. Please try again.';

/*
Retry logic for 504 errors
*/
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

/*
Interceptor for better error handling
*/
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

/*
Special functions for different API calls with adjusted timeouts
*/
export const apiCall = {  
  quick: (config) => axios({ ...config, timeout: 30000 }),
  normal: (config) => axios({ ...config, timeout: 120000 }),
  ml: (config) => retryRequest({ ...config, timeout: 600000 }),
  batch: (config) => retryRequest({ ...config, timeout: 900000 }), // 15 Minutes
};

/*
Fallback CORS-Proxy (if the API proxy has CORS problems)
*/
// export const API_BASE_URL = "https://cors-anywhere.herokuapp.com/https://masterthesis-api-proxy.onrender.com";

/*
Alternative CORS-Proxy
*/
// export const API_BASE_URL = "https://api.allorigins.win/raw?url=https://masterthesis-api-proxy.onrender.com";

/*
Local development (commented out)
*/
// export const API_BASE_URL = "http://localhost:8080";

/*
API Proxy (commented out because of CORS problems)
*/
// export const API_BASE_URL = "https://masterthesis-api-proxy.onrender.com";

/*
Cloudflare Tunnel (for local testing)
*/
// export const API_BASE_URL = "https://lot-realtors-tribune-shapes.trycloudflare.com"; 