# Master Thesis

"A Machine Learning Approach to Anticipate the Timing of Carrer Progressions"

---

**Frontend**: https://masterthesis-igbq.onrender.com
**Backend**: https://masterthesis-backend.onrender.com

---

## Overview

This application analyzes LinkedIn profiles and predicts with the help of AI models when a career step is likely to occur. It consists of a **Backend** (Python/Flask, AI models) and a **Frontend** (React, Material-UI), both deployed on Render.

---

## Deployment Status

- **Frontend**: Live on Render (https://masterthesis-igbq.onrender.com)
- **Backend**: Live on Render (https://masterthesis-backend.onrender.com)
- **ML Models**: All 3 models (TFT, GRU, XGBoost) are integrated in the backend

---

## Project Structure (with explanations)

```
.
├── backend/                 # Backend server (Python, Flask, AI models)
│   ├── app.py               # Main API server (Flask) - Render optimized
│   ├── config.py            # Configuration (CORS, database, model paths)
│   ├── requirements.txt     # Python dependencies
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
│   │   ├── api.js           # API URL configuration (Render backend)
│   │   └── ...
│   ├── public/              # Static files (HTML, Icons, ...)
│   ├── package.json         # Frontend dependencies
│   └── tailwind.config.js   # Tailwind CSS configuration
│
├── README.md                # This guide
└── ...
```

**Important notes:**
- **Backend**: All ML models, data processing, API logic - deployed on Render
- **Frontend**: User interface, communicates with Render backend

---

### Frontend
- **LinkedIn Profile Analysis**: Analyze individual profiles and calculate switching probability
- **Batch Upload**: Upload multiple profiles via CSV and analyze them
- **Candidate Management**: Search, filter, and save results
- **Admin Area**: Create, edit, delete users, manage permissions
- **Explanations**: SHAP-based feature importance for each prediction

### Backend
- **API Server**: REST API for all frontend functions (deployed on Render)
- **AI Models**: Temporal Fusion Transformer (TFT), GRU, XGBoost (all integrated)
- **Feature Engineering**: Automatic preparation and transformation of LinkedIn data
- **Database**: Storage of candidates and analyses in MongoDB
- **Explainability**: SHAP & Lime integration for comprehensible predictions

---

## Data Flow & Processed Data

1. **Frontend** (Render): User enters LinkedIn profile URL or CSV
2. **API Call**: Frontend sends data to backend (https://masterthesis-backend.onrender.com)
3. **Backend** (Render): Prepares data, applies ML model
4. **Explanation**: SHAP calculates the most important influencing factors
5. **Response**: Backend sends prediction & explanations to frontend
6. **Frontend**: Displays results, visualizations and explanations

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
  - **Deployment**: Render (Web Service)
- **Frontend:**
  - React, Material-UI, Tailwind CSS
  - React Router
  - **Deployment**: Render (Static Site)
- **DevOps:**
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

4. **Adjust API URL for local development**
   ```bash
   # In frontend/src/api.js
   export const API_BASE_URL = "http://localhost:8080";
   ```

5. **Start application**
   ```bash
   # Backend (Terminal 1)
   cd backend
   python app.py
   
   # Frontend (Terminal 2)
   cd frontend
   npm start
   ```

### Option 2: Render Deployment (Production)

**Frontend & Backend are already deployed and running 24/7:**

- **Frontend**: https://masterthesis-igbq.onrender.com
- **Backend**: https://masterthesis-backend.onrender.com
- **No local installation needed**

---

## API Overview (most important endpoints)

| Method | Path                | Description                        | Status |
|---------|---------------------|-------------------------------------|---------|
| POST    | /scrape-linkedin    | Analyze LinkedIn profile            | ✅ Live |
| POST    | /predict            | Prediction for one profile          | ✅ Live |
| POST    | /predict-batch      | Batch prediction for multiple profiles | ✅ Live |
| GET     | /candidates         | Get all saved candidates            | ✅ Live |
| POST    | /api/candidates     | Save candidates                     | ✅ Live |
| GET     | /api/users          | User list (Admin)                   | ✅ Live |
| POST    | /api/create-user    | Create new user (Admin)             | ✅ Live |
| PUT     | /api/users/:id      | Edit user (Admin)                   | ✅ Live |
| DELETE  | /api/users/:id      | Delete user (Admin)                 | ✅ Live |

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
- **Deployment**: Automatically via GitHub → Render

### Frontend Development
- **New pages**: Add in `frontend/src/pages/`
- **API URL**: Adjust in `frontend/src/api.js`
- **UI/UX**: Extend with Material-UI & Tailwind
- **Deployment**: Automatically via GitHub → Render

### Render Deployment
- **Automatic deployments** on GitHub pushes
- **Health checks** for backend monitoring
- **Logs** available for debugging
- **Scaling** possible when needed

---

## Troubleshooting

### Common Issues
1. **CORS errors**: Check CORS origins in `backend/app.py`
2. **Import errors**: All imports use relative paths
3. **Model loading**: First prediction takes longer
4. **Memory limits**: Render Free Plan has 512MB RAM

### View Logs
- **Render Dashboard**: Service → Logs
- **Backend**: Automatic logs in Render
- **Frontend**: Browser Developer Tools

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
