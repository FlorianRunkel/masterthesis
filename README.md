# Master Thesis

"Anticipating Job Changes: An Explainable Machine Learning Approach for Talent Acquisition"

---

- **Frontend**: https://masterthesis-igbq.onrender.com
- **API Proxy**: https://masterthesis-api-proxy.onrender.com
- **Backend**: AWS ECS (Elastic Container Service)

---
## Abstract

<div align="justify">

This thesis examines the challenge of forecasting candidate mobility in Active Recruiting, an issue of growing importance in Human Resource Management. While prior research has largely focused on binary attrition prediction, the temporal dynamics of career changes and the explainability of models remain underexplored. To address this gap, a web-based artefact was developed within the Design Science Research paradigm, combining machine learning with explainable AI. 

The framework applies XGBoost to classify candidates’ willingness to change jobs and sequential models, including Gated Recurrent Units and the Temporal Fusion Transformer, to forecast the timing of potential transitions. Explainability techniques such as SHAP and LIME were integrated to provide global and local transparency. Evaluation followed a twofold approach, comprising technical performance analysis and a user study. Results show that XGBoost achieves high predictive accuracy and recall, while the Temporal Fusion Transformer outperforms GRU in precision and practical relevance. The user study further demonstrates that explainability enhances transparency, usability, and trust, thereby strengthening the acceptance of predictive systems in recruiting. 

Overall, this thesis demonstrates that integrating accurate prediction models with explainable AI enables robust classification and temporal forecasting of candidate mobility while simultaneously enhancing interpretability and practical applicability. 

Keywords: Human Resource Management, Active Recruiting, Temporal Forecasting of Career Changes, Machine Learning (ML), explainable artificial itelligence (XAI)

</div>
---

## Project Structure

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
├── README.md                # This guide
└── ...

masterthesis-api-proxy/          # API Proxy server (Node.js, Express)
├── server.js                    # Proxy server configuration
├── package.json                 # Proxy dependencies
└── ...
```

---

### Frontend
- **Manual Input**: Enter candidate data manually for detailed career predictions
- **LinkedIn Profile Analysis**: Analyze individual LinkedIn profiles and calculate switching probability
- **Batch Upload**: Upload multiple profiles via CSV and analyze them in bulk
- **Candidate Management**: Search, filter, and save promising candidates for future contact
- **Admin Area**: Create, edit, delete users and manage system permissions
- **Explanations**: SHAP-based feature importance and explainable AI insights for each prediction

### Backend
- **API Server**: REST API for all frontend functions (deployed on AWS ECS)
- **AI Models**: Temporal Fusion Transformer (TFT), GRU, XGBoost (all integrated)
- **Feature Engineering**: Automatic preparation and transformation of LinkedIn data
- **Database**: Storage of candidates and analyses in MongoDB
- **Explainability**: SHAP & Lime integration for comprehensible predictions

---

## Data Flow & Processed Data

1. **Frontend** (Render): User enters LinkedIn profile URL or CSV
2. **API Call**: Frontend sends data to API Proxy (https://masterthesis-api-proxy.onrender.com)
3. **API Proxy** (Render): Handles CORS and forwards request to AWS backend
4. **Backend** (AWS ECS): Prepares data, applies ML model
5. **Explanation**: SHAP/Lime calculates the most important influencing factors
6. **Response**: Backend sends prediction & explanations back through proxy to frontend
7. **Frontend**: Displays results, visualizations and explanations

---

## Technologies

- **Backend:**
  - Python 3.x, Flask, Flask-CORS
  - PyTorch (TFT, GRU), XGBoost
  - SHAP & Lime (Explainable AI)
  - MongoDB (Database)
- **Frontend:**
  - React, Material-UI, Tailwind CSS
  - React Router
- **API Proxy:**
  - Node.js, Express, http-proxy

---

## API Overview

| Method | Path                | Description
|---------|---------------------|-------------------------------------
| POST    | /api/scrape-linkedin    | Analyze LinkedIn profile
| POST    | /api/predict            | Prediction for one profile
| POST    | /api/predict-batch      | Batch prediction for multiple profiles
| POST    | /api/login              | User authentication
| GET     | /api/candidates         | Get all saved candidates
| POST    | /api/candidates         | Save candidates
| GET     | /api/users              | User list (Admin)
| POST    | /api/create-user        | Create new user (Admin)
| PUT     | /api/users/:id          | Edit user (Admin)
| DELETE  | /api/users/:id          | Delete user (Admin)

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
- **Batch processing**: Supports up to 20 profiles simultaneously

**Dependencies:**
- **XGBoost**: 3.0.0 (latest stable version)
- **PyTorch**: 2.1.0 (stable version for production)
- **PyTorch Lightning**: 2.1.0 (compatible with PyTorch 2.1.0)

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

## Author & Contact

**Florian Runkel**

**Email**: runkel.florian@stud.uni-regensburg.de

**University**: University of Regensburg

**Project**: Master Thesis

---

## License

This project is part of a Master's thesis at the University of Regensburg. All rights reserved.

---
