# Pneumonia XAI Fullstack — Project Documentation

> **Claude-এর সাথে কাজ করার সময় এই file টা share করো।**
> "এই PROJECT.md দেখো, তারপর X করো" বললেই Claude পুরো project বুঝে কাজ করবে।

GitHub: https://github.com/arafattech/pneumonia-xai-fullstack

---

## Project কী করে

Chest X-ray image upload করলে:
1. AI বলে NORMAL না PNEUMONIA
2. Grad-CAM heatmap দেখায় — model কোন জায়গা দেখে সিদ্ধান্ত নিল
3. LIME explanation দেখায় — কোন pixel গুলো গুরুত্বপূর্ণ ছিল
4. প্রতিটা prediction MySQL-এ save হয়

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| AI Model | DenseNet121 (PyTorch) |
| XAI | Grad-CAM + LIME |
| Backend | FastAPI (Python 3.12) |
| Database | MySQL 8 (SQLAlchemy + PyMySQL) |
| Frontend | React + Vite |
| Dataset | Kaggle Chest X-Ray Pneumonia |

---

## Folder Structure

```
pneumonia_xai_fullstack/
├── backend/                    # FastAPI server
│   ├── main.py                 # App entry point, startup hooks
│   ├── requirements.txt        # Python dependencies
│   ├── venv/                   # Python virtual env (git ignored)
│   └── app/
│       ├── api/
│       │   ├── predict.py      # POST /predict → prediction + DB save
│       │   ├── xai.py          # POST /explain/gradcam, /explain/lime
│       │   └── health.py       # GET /health
│       ├── core/
│       │   ├── config.py       # All settings (model, DB, image)
│       │   └── model_loader.py # Model load on startup (singleton)
│       ├── db/
│       │   ├── database.py     # SQLAlchemy engine + session
│       │   └── models.py       # Prediction table schema
│       ├── ml/
│       │   ├── model.py        # DenseNet121 build/load
│       │   ├── gradcam.py      # Grad-CAM implementation
│       │   ├── lime_exp.py     # LIME implementation
│       │   └── weights/        # best_model.pth রাখার জায়গা (git ignored)
│       ├── schemas/
│       │   └── schemas.py      # Pydantic request/response models
│       └── utils/
│           └── image_utils.py  # Image validation + preprocessing
│
├── frontend/                   # React app
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Home.jsx        # Image upload + prediction result
│   │   │   ├── Explain.jsx     # Grad-CAM + LIME visualization
│   │   │   └── About.jsx       # Project info
│   │   ├── components/
│   │   │   ├── UploadCard.jsx  # Drag-drop image uploader
│   │   │   ├── ResultCard.jsx  # Prediction result display
│   │   │   ├── HeatmapViewer.jsx # Heatmap image display
│   │   │   ├── Navbar.jsx      # Navigation
│   │   │   └── LoadingSpinner.jsx
│   │   ├── hooks/
│   │   │   └── usePredict.js   # API call + state management hook
│   │   └── utils/
│   │       └── api.js          # Axios client (BASE_URL: VITE_API_URL)
│   ├── vite.config.js
│   └── package.json
│
├── ml_pipeline/                # Model training
│   ├── run_pipeline.py         # Entry point — train করতে এটা run করো
│   ├── requirements.txt        # ML-specific deps
│   └── src/
│       ├── config.py           # Paths, hyperparameters (env var override)
│       ├── dataset.py          # DataLoader + augmentation
│       ├── model.py            # DenseNet121 / ResNet50 / EfficientNet
│       ├── train.py            # Training loop
│       ├── evaluate.py         # Metrics, confusion matrix
│       ├── eda.py              # Dataset statistics
│       └── xai.py              # Grad-CAM + LIME + SHAP (training-time)
│
├── docs/
│   ├── PROJECT.md              # এই file (master reference)
│   ├── API.md                  # API endpoints detail
│   └── DATASET_SETUP.md        # Kaggle dataset setup steps
│
├── run.sh                      # App চালানোর script
├── setup_data.sh               # Data download + train + deploy script
├── docker-compose.yml          # Docker setup (backend:8000, frontend:5173)
└── .gitignore
```

---

## Key Files — কোনটা কী করে

### `backend/app/core/config.py`
সব settings এক জায়গায়। Change করতে হলে এখানে অথবা `.env` file-এ:

```python
DB_HOST     = "localhost"
DB_PORT     = 3306
DB_USER     = "root"
DB_PASSWORD = "root"
DB_NAME     = "pneumonia_xai"

MODEL_PATH  = "backend/app/ml/weights/best_model.pth"
MODEL_NAME  = "densenet121"   # densenet121 | resnet50 | efficientnet_b0
DEVICE      = "cpu"           # "cuda" GPU থাকলে
IMG_SIZE    = 224
MAX_FILE_SIZE = 10MB
```

### `backend/app/db/models.py`
MySQL `predictions` table:

| Column | Type | Description |
|--------|------|-------------|
| id | INT PK | Auto increment |
| filename | VARCHAR(255) | Uploaded file name |
| predicted_class | VARCHAR(50) | NORMAL / PNEUMONIA |
| predicted_index | INT | 0 = NORMAL, 1 = PNEUMONIA |
| confidence | FLOAT | 0.0 – 1.0 |
| prob_normal | FLOAT | NORMAL probability |
| prob_pneumonia | FLOAT | PNEUMONIA probability |
| model_name | VARCHAR(100) | densenet121 |
| created_at | DATETIME | Auto timestamp |

### `ml_pipeline/src/config.py`
Training hyperparameters। ENV var দিয়ে override করা যায়:

```bash
DATA_ROOT="/path/to/chest_xray"   # dataset location
SAVE_DIR="/path/to/save/models"   # model output location
```

Default: `ml_pipeline/data/chest_xray/` এবং `ml_pipeline/models/`

---

## Data Flow

```
User uploads image
       ↓
frontend/src/utils/api.js  →  POST /predict
       ↓
backend/app/api/predict.py
  ├── validate_image()       # type + size check
  ├── bytes_to_pil()         # PIL Image
  ├── pil_to_tensor()        # normalize + resize to 224×224
  ├── model(img_tensor)      # DenseNet121 forward pass
  ├── softmax → probs        # [prob_normal, prob_pneumonia]
  └── db.add(Prediction())   # MySQL-এ save
       ↓
JSON response → ResultCard.jsx
```

---

## API Endpoints

| Method | URL | কী করে |
|--------|-----|--------|
| GET | `/health` | Server + model status |
| POST | `/predict` | Image upload → prediction + DB save |
| POST | `/explain/gradcam` | Grad-CAM heatmap (base64 PNG) |
| POST | `/explain/lime` | LIME superpixel explanation |

Full detail: [API.md](API.md)

---

## App চালানো

### Local (Docker ছাড়া)
```bash
# প্রথমবার — deps install
cd backend && python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt && deactivate

cd ../frontend && npm install

# App চালাও
./run.sh              # backend + frontend দুটোই
./run.sh backend      # শুধু backend
./run.sh frontend     # শুধু frontend
```

### Docker
```bash
./run.sh docker
```

### URLs
- Backend API: http://localhost:8000
- API Docs (Swagger): http://localhost:8000/docs
- Frontend: http://localhost:5173

---

## Model Training

Full guide: [DATASET_SETUP.md](DATASET_SETUP.md)

```bash
# Shortcut — সব একসাথে
./setup_data.sh

# Manual
cd ml_pipeline
DATA_ROOT="$(pwd)/data/chest_xray" SAVE_DIR="$(pwd)/models" \
  python3 run_pipeline.py --skip-eda --epochs 5

# Weights backend-এ copy
cp ml_pipeline/models/best_model.pth backend/app/ml/weights/
```

Training args:
```
--skip-eda      EDA skip (faster)
--skip-train    Training skip, শুধু evaluate
--eval-only     Saved model load করে evaluate
--skip-xai      XAI visualization skip
--epochs N      Epoch count (default: 5)
--model NAME    densenet121 | resnet50 | efficientnet_b0
```

---

## Database

```sql
-- MySQL connect
mysql -u root -proot pneumonia_xai

-- Recent predictions দেখো
SELECT id, filename, predicted_class, confidence, created_at
FROM predictions
ORDER BY created_at DESC
LIMIT 10;

-- Stats
SELECT predicted_class, COUNT(*), AVG(confidence)
FROM predictions
GROUP BY predicted_class;
```

---

## Environment Variables (.env file)

`backend/` folder-এ `.env` file বানিয়ে override করো:

```env
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=root
DB_NAME=pneumonia_xai

MODEL_PATH=/path/to/best_model.pth
MODEL_NAME=densenet121
DEVICE=cpu

DEBUG=false
```

Frontend-এ `frontend/.env`:
```env
VITE_API_URL=http://localhost:8000
```

---

## Common Tasks — Claude-কে বলার ভাষা

| কী করতে চাও | Claude-কে বলো |
|-------------|--------------|
| নতুন API endpoint যোগ | "PROJECT.md দেখো। `/predict` এর মতো `/batch-predict` endpoint বানাও যেটা একসাথে multiple image নেবে" |
| DB table বদলাও | "PROJECT.md দেখো। predictions table-এ `patient_id` column যোগ করো" |
| নতুন model support | "PROJECT.md দেখো। config.py-তে VGG16 support যোগ করো" |
| Frontend page | "PROJECT.md দেখো। History page বানাও যেটা DB থেকে past predictions দেখাবে" |
| Training hyperparameter বদলাও | "PROJECT.md দেখো। ml_pipeline/src/config.py-তে batch size 16 করো আর epochs 25 করো" |

---

## Troubleshoot

| Error | কারণ | Fix |
|-------|------|-----|
| `Model not found` | weights নাই | `setup_data.sh` চালাও অথবা weights copy করো |
| `Can't connect to MySQL` | MySQL চলছে না | `sudo systemctl start mysql` |
| `Access denied for user root` | MySQL password | `ALTER USER 'root'@'localhost' IDENTIFIED BY 'root';` |
| `Module not found` | venv activate করোনি | `source backend/venv/bin/activate` |
| `CORS error` | Frontend URL mismatch | `config.py` → `ALLOWED_ORIGINS` চেক করো |
| `413 Request Entity Too Large` | File > 10MB | ছোট image দাও |

---

## Dependencies

### Backend (requirements.txt)
```
fastapi==0.111.0
uvicorn[standard]==0.29.0
torch==2.3.0 + torchvision==0.18.0
Pillow, opencv-python-headless
scikit-learn, lime, scipy
sqlalchemy==2.0.30, pymysql==1.1.1
pydantic-settings==2.2.1
```

### Frontend (package.json)
```
react, react-dom, react-router-dom
axios
vite
```

### ML Pipeline (ml_pipeline/requirements.txt)
```
torch, torchvision
matplotlib, seaborn
scikit-learn, lime, shap (optional)
pandas, numpy
```
