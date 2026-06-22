# 🫁 Pneumonia XAI — Full Stack Project

**Explainable AI-based Pneumonia Detection** from Chest X-ray Images  
Stack: **React + FastAPI + PyTorch + Grad-CAM + LIME**

---

## 📁 Project Structure

```
pneumonia_xai_fullstack/
│
├── backend/                   ← FastAPI Python backend
│   ├── app/
│   │   ├── api/               ← Route handlers
│   │   │   ├── predict.py     ← /predict endpoint
│   │   │   ├── xai.py         ← /explain (Grad-CAM, LIME)
│   │   │   └── health.py      ← /health endpoint
│   │   ├── core/
│   │   │   ├── config.py      ← App settings
│   │   │   └── model_loader.py← Singleton model loader
│   │   ├── ml/
│   │   │   ├── model.py       ← DenseNet121 builder
│   │   │   ├── gradcam.py     ← Grad-CAM implementation
│   │   │   └── lime_exp.py    ← LIME implementation
│   │   ├── schemas/
│   │   │   └── schemas.py     ← Pydantic request/response models
│   │   └── utils/
│   │       └── image_utils.py ← Image preprocessing helpers
│   ├── main.py                ← FastAPI app entry point
│   ├── requirements.txt
│   └── Dockerfile
│
├── frontend/                  ← React frontend
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── components/        ← Reusable UI components
│   │   │   ├── Navbar.jsx
│   │   │   ├── UploadCard.jsx
│   │   │   ├── ResultCard.jsx
│   │   │   ├── HeatmapViewer.jsx
│   │   │   ├── MetricsBar.jsx
│   │   │   └── LoadingSpinner.jsx
│   │   ├── pages/
│   │   │   ├── Home.jsx       ← Upload & predict page
│   │   │   ├── Explain.jsx    ← XAI heatmap page
│   │   │   └── About.jsx      ← Project info page
│   │   ├── hooks/
│   │   │   └── usePredict.js  ← API call hooks
│   │   ├── utils/
│   │   │   └── api.js         ← Axios API client
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── package.json
│   └── vite.config.js
│
├── ml_pipeline/               ← Standalone Kaggle training pipeline
│   ├── src/
│   │   ├── config.py
│   │   ├── eda.py
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── xai.py
│   └── run_pipeline.py        ← Single entry point
│
├── docs/
│   └── API.md                 ← API documentation
│
├── docker-compose.yml         ← Run everything together
└── README.md
```

---

## 🚀 Quick Start

### Option 1: Docker (easiest)
```bash
docker-compose up --build
# Frontend → http://localhost:5173
# Backend  → http://localhost:8000
# API Docs → http://localhost:8000/docs
```

### Option 2: Manual

**Backend:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

**ML Training (Kaggle):**
```bash
cd ml_pipeline
pip install -r requirements.txt
python run_pipeline.py
```

---

## 🔗 Dataset

Kaggle: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Place trained model at: `backend/app/ml/weights/best_model.pth`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Predict from X-ray image |
| POST | `/explain/gradcam` | Grad-CAM heatmap |
| POST | `/explain/lime` | LIME explanation |
| GET | `/docs` | Swagger UI |
