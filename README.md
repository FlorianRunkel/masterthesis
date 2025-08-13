# Master Thesis

"A Machine Learning Approach to Anticipate the Timing of Carrer Progressions"

---

**Frontend**: https://masterthesis-igbq.onrender.com
**API Proxy**: https://masterthesis-api-proxy.onrender.com
**Backend**: AWS ECS (Elastic Container Service)

---

## Overview

This application analyzes LinkedIn profiles and predicts with the help of AI models when a career step is likely to occur. It consists of a **Backend** (Python/Flask, AI models) deployed on AWS, a **Frontend** (React, Material-UI) deployed on Render, and an **API Proxy** (Node.js/Express) also deployed on Render to handle CORS and routing.

---

## Deployment Status

- **Frontend**: Live on Render (https://masterthesis-igbq.onrender.com)
- **API Proxy**: Live on Render (https://masterthesis-api-proxy.onrender.com)
- **Backend**: Live on AWS ECS (Elastic Container Service)
- **ML Models**: All 3 models (TFT, GRU, XGBoost) are integrated in the backend

---

## Project Structure (with explanations)

```
.
├── backend/                 # Backend server (Python, Flask, AI models)
│   ├── app.py               # Main API server (Flask) - AWS optimized
│   ├── config.py            # Configuration (CORS, database, model paths)
│   ├── requirements.txt     # Python dependencies
│   ├── Dockerfile           # Docker configuration for AWS deployment
│   ├── ml_pipe/             # Machine Learning Pipeline
│   │   ├── data/            # Data processing & Feature Engineering
│   │   │   ├── featureEngineering/   # Feature engineering scripts
│   │   │   └── database/             # Database connection (MongoDB)
│   │   ├── explainable_ai/  # SHAP & Lime Explainability
│   │   └── models/          # AI models (TFT, GRU, XGBoost)
│   │       ├── tft/saved_models/     # TFT model (39MB)
│   │       ├── gru/saved_models/     # GRU model (30MB)
│   │       └── xgboost/saved_models/ # XGBoost model (580KB)
│   └── api_handlers/        # API handlers for various endpoints
│
├── frontend/                # React Frontend (User interface)
│   ├── src/                 # React components & logic
│   │   ├── pages/           # Pages (e.g., Analysis, Admin, Candidates)
│   │   ├── api.js           # API URL configuration (API Proxy)
│   │   └── ...
│   ├── public/              # Static files (HTML, Icons, ...)
│   ├── package.json         # Frontend dependencies
│   └── tailwind.config.js   # Tailwind CSS configuration
│
├── masterthesis-api-proxy/  # API Proxy server (Node.js, Express)
│   ├── server.js            # Proxy server configuration
│   ├── package.json         # Proxy dependencies
│   └── ...
│
├── README.md                # This guide
└── ...
```

**Important notes:**
- **Backend**: All ML models, data processing, API logic - deployed on AWS ECS
- **Frontend**: User interface, communicates with API Proxy
- **API Proxy**: Handles CORS, routing, and forwards requests to AWS backend

---

### Frontend
- **LinkedIn Profile Analysis**: Analyze individual profiles and calculate switching probability
- **Batch Upload**: Upload multiple profiles via CSV and analyze them
- **Candidate Management**: Search, filter, and save results
- **Admin Area**: Create, edit, delete users, manage permissions
- **Explanations**: SHAP-based feature importance for each prediction

### Backend
- **API Server**: REST API for all frontend functions (deployed on AWS ECS)
- **AI Models**: Temporal Fusion Transformer (TFT), GRU, XGBoost (all integrated)
- **Feature Engineering**: Automatic preparation and transformation of LinkedIn data
- **Database**: Storage of candidates and analyses in MongoDB
- **Explainability**: SHAP & Lime integration for comprehensible predictions

### API Proxy
- **CORS Handling**: Manages cross-origin requests from frontend
- **Request Routing**: Forwards all requests to AWS backend
- **Load Balancing**: Acts as intermediary between frontend and backend

---

## Data Flow & Processed Data

1. **Frontend** (Render): User enters LinkedIn profile URL or CSV
2. **API Call**: Frontend sends data to API Proxy (https://masterthesis-api-proxy.onrender.com)
3. **API Proxy** (Render): Handles CORS and forwards request to AWS backend
4. **Backend** (AWS ECS): Prepares data, applies ML model
5. **Explanation**: SHAP calculates the most important influencing factors
6. **Response**: Backend sends prediction & explanations back through proxy to frontend
7. **Frontend**: Displays results, visualizations and explanations

**Processed Data:**
- LinkedIn profiles (manually or via CSV)
- Feature vectors (extracted from profiles)
- Prediction results & SHAP values
- User management (admin area)

---

## Technologies

- **Backend:**
  - Python 3.x, Flask, Flask-CORS
  - PyTorch (TFT, GRU), XGBoost
  - SHAP & Lime (Explainable AI)
  - MongoDB (Database)
  - **Deployment**: AWS ECS (Elastic Container Service)
- **Frontend:**
  - React, Material-UI, Tailwind CSS
  - React Router
  - **Deployment**: Render (Static Site)
- **API Proxy:**
  - Node.js, Express, http-proxy
  - **Deployment**: Render (Web Service)
- **DevOps:**
  - AWS ECR (Elastic Container Registry)
  - AWS ECS (Elastic Container Service)
  - Render
  - GitHub (code versioning)

---

## Setup & Deployment

### Option 1: Local Development (for developers)

1. **Clone repository**
   ```bash
   git clone [repository-url]
   cd [repository-name]
   ```

2. **Install backend dependencies**
   ```bash
   cd backend
   pip3 install -r requirements.txt
   ```

3. **Install frontend dependencies**
   ```bash
   cd ../frontend
   npm install
   ```

4. **Install API proxy dependencies**
   ```bash
   cd ../masterthesis-api-proxy
   npm install
   ```

5. **Adjust API URL for local development**
   ```bash
   # In frontend/src/api.js
   export const API_BASE_URL = "http://localhost:10000";
   ```

6. **Start application**
   ```bash
   # Backend (Terminal 1)
   cd backend
   python app.py
   
   # API Proxy (Terminal 2)
   cd masterthesis-api-proxy
   node server.js
   
   # Frontend (Terminal 3)
   cd frontend
   npm start
   ```

### Option 2: Render Deployment (Production)

**Frontend & API Proxy are already deployed and running 24/7:**

- **Frontend**: https://masterthesis-igbq.onrender.com
- **API Proxy**: https://masterthesis-api-proxy.onrender.com
- **Backend**: AWS ECS (Elastic Container Service)

### Option 3: AWS Deployment (Production)

**Backend deployment to AWS ECS:**

1. **Build Docker Image:**
   ```bash
   cd backend
   docker build --platform linux/amd64 --no-cache -t masterthesis-backend .
   ```

2. **Tag Image for AWS ECR:**
   ```bash
   docker tag masterthesis-backend:latest 736874312333.dkr.ecr.eu-central-1.amazonaws.com/masterthesis-backend:latest
   ```

3. **Login to AWS ECR:**
   ```bash
   aws ecr get-login-password --region eu-central-1 | docker login --username AWS --password-stdin 736874312333.dkr.ecr.eu-central-1.amazonaws.com
   ```

4. **Push to AWS ECR:**
   ```bash
   docker push 736874312333.dkr.ecr.eu-central-1.amazonaws.com/masterthesis-backend:latest
   ```

**Note**: The AWS ECR repository URL (`736874312333.dkr.ecr.eu-central-1.amazonaws.com`) is specific to this deployment. Update the URL in your configuration if using a different AWS account.

---

## API Overview (most important endpoints)

| Method | Path                | Description                        | Status |
|---------|---------------------|-------------------------------------|---------|
| POST    | /api/scrape-linkedin    | Analyze LinkedIn profile            | ✅ Live |
| POST    | /api/predict            | Prediction for one profile          | ✅ Live |
| POST    | /api/predict-batch      | Batch prediction for multiple profiles | ✅ Live |
| POST    | /api/login              | User authentication                 | ✅ Live |
| GET     | /api/candidates         | Get all saved candidates            | ✅ Live |
| POST    | /api/candidates         | Save candidates                     | ✅ Live |
| GET     | /api/users              | User list (Admin)                   | ✅ Live |
| POST    | /api/create-user        | Create new user (Admin)             | ✅ Live |
| PUT     | /api/users/:id          | Edit user (Admin)                   | ✅ Live |
| DELETE  | /api/users/:id          | Delete user (Admin)                 | ✅ Live |

**Note**: All API endpoints now use the `/api` prefix for consistency.

---

## ML Models & Performance

**Integrated Models:**
- **TFT (Temporal Fusion Transformer)**: 39MB - Time series predictions
- **GRU (Gated Recurrent Unit)**: 30MB - Sequence-based predictions  
- **XGBoost**: 580KB - Classification & regression

**Performance:**
- **First prediction**: 5-10 seconds (model loading)
- **Subsequent predictions**: 1-3 seconds
- **Batch processing**: Supports up to 100 profiles simultaneously

**Dependencies:**
- **XGBoost**: 3.0.0 (latest stable version)
- **PyTorch**: 2.1.0 (stable version for production)
- **PyTorch Lightning**: 2.1.0 (compatible with PyTorch 2.1.0)

---

## Example Workflow (Use Case)

1. **Admin creates user** (in admin area)
2. **User uploads LinkedIn profile** (manually or via CSV)
3. **Selects model & starts analysis**
4. **Receives prediction & explanations** (e.g., feature importance)
5. **Saves interesting candidates**
6. **Admin can manage users, assign permissions, etc.**

---

## Development & Customization

### Backend Development
- **New models**: Add in `backend/ml_pipe/models/`
- **Feature engineering**: In `backend/ml_pipe/data/featureEngineering/`
- **API logic**: In `backend/app.py` and `backend/api_handlers/`
- **Deployment**: Docker build → AWS ECR → AWS ECS

### Frontend Development
- **New pages**: Add in `frontend/src/pages/`
- **API URL**: Adjust in `frontend/src/api.js`
- **UI/UX**: Extend with Material-UI & Tailwind
- **Deployment**: Automatically via GitHub → Render

### API Proxy Development
- **Routing logic**: Modify `masterthesis-api-proxy/server.js`
- **CORS settings**: Adjust in proxy server
- **Deployment**: Automatically via GitHub → Render

### AWS Deployment
- **Docker builds**: Use `--platform linux/amd64 --no-cache`
- **ECR management**: Tag and push to correct repository
- **ECS updates**: Automatic deployment from ECR

---

## Troubleshooting

### Common Issues
1. **CORS errors**: Check CORS origins in API proxy server
2. **Import errors**: All imports use relative paths
3. **Model loading**: First prediction takes longer
4. **Docker build errors**: Ensure `--platform linux/amd64` for AWS compatibility
5. **XGBoost errors**: Ensure `requirements.txt` has `xgboost==3.0.0`

### View Logs
- **Render Dashboard**: Service → Logs (Frontend & API Proxy)
- **AWS ECS**: CloudWatch logs for backend
- **Frontend**: Browser Developer Tools

### Docker Cleanup
If you encounter "No space left on device" errors:
```bash
# Clean up Docker system
docker system prune -a --volumes

# Rebuild with clean cache
docker build --platform linux/amd64 --no-cache -t masterthesis-backend .
```

---

## Author & Contact

**Florian Runkel**  
**Email**: runkel.florian@stud.uni-regensburg.de  
**University**: University of Regensburg  
**Project**: Master's Thesis - LinkedIn Career Prediction AI

---

## License

This project is part of a Master's thesis at the University of Regensburg. All rights reserved.

---
