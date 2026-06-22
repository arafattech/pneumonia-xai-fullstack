# Full Setup Guide — New PC থেকে শুরু করলে

এই guide follow করলে নতুন PC-তে শুরু থেকে সব কাজ করবে।

---

## Prerequisites — System Requirements

| Tool | Version | Check Command |
|------|---------|---------------|
| Python | 3.10+ | `python3 --version` |
| Node.js | 18+ | `node --version` |
| npm | 9+ | `npm --version` |
| MySQL | 8.0 | `mysql --version` |
| Git | any | `git --version` |

---

## Step 1 — System Dependencies Install

### Ubuntu / Debian
```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv nodejs npm git curl

# MySQL
sudo apt install -y mysql-server
sudo systemctl start mysql
sudo systemctl enable mysql
```

### MySQL root password set করো (fresh install হলে)
```bash
sudo mysql
```
MySQL prompt-এ:
```sql
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'root';
FLUSH PRIVILEGES;
EXIT;
```

---

## Step 2 — Project Clone করো

```bash
cd ~/Downloads
git clone https://github.com/arafattech/pneumonia-xai-fullstack.git pneumonia_xai_fullstack
cd pneumonia_xai_fullstack
```

---

## Step 3 — MySQL Database তৈরি করো

```bash
mysql -u root -proot -e "CREATE DATABASE IF NOT EXISTS pneumonia_xai CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
```

Verify:
```bash
mysql -u root -proot -e "SHOW DATABASES;" | grep pneumonia
# Output: pneumonia_xai
```

---

## Step 4 — Backend Setup

```bash
cd backend

# Virtual environment তৈরি করো
python3 -m venv venv
source venv/bin/activate

# Dependencies install করো
pip install -r requirements.txt

# DB tables তৈরি করো
python3 -c "from app.db.database import init_db; init_db(); print('DB tables created')"

deactivate
cd ..
```

---

## Step 5 — Frontend Setup

```bash
cd frontend
npm install
cd ..
```

---

## Step 6 — Kaggle Dataset Setup

### 6a. Kaggle Account API Token বানাও
1. Browser: `https://www.kaggle.com/settings/account`
2. **API** section → **"Create New Token"** click
3. `kaggle.json` download হবে (অথবা access token দেখাবে)

### 6b-i. যদি `kaggle.json` পাও (পুরনো format)
```bash
mkdir -p ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 6b-ii. যদি শুধু token দেখায় (নতুন KGAT_ format)
```bash
mkdir -p ~/.kaggle
echo -n "KGAT_xxxxxxxxxxxxxxxxxxxx" > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

### 6c. Kaggle CLI install + test
```bash
source backend/venv/bin/activate
pip install kaggle
kaggle datasets list | head -3   # কাজ করলে credentials OK
deactivate
```

### 6d. Dataset download করো (~2GB, 10-15 min)
```bash
mkdir -p ml_pipeline/data
cd ml_pipeline/data

source ../../backend/venv/bin/activate
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia --unzip
deactivate

cd ../..
```

Download শেষে structure হবে:
```
ml_pipeline/data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/       (1341 images)
    │   └── PNEUMONIA/    (3875 images)
    ├── val/
    │   ├── NORMAL/       (8 images)
    │   └── PNEUMONIA/    (8 images)
    └── test/
        ├── NORMAL/       (234 images)
        └── PNEUMONIA/    (390 images)
```

Nested folder হলে fix করো:
```bash
# যদি chest_xray/chest_xray/ double থাকে
mv ml_pipeline/data/chest_xray/chest_xray/* ml_pipeline/data/chest_xray/
rmdir ml_pipeline/data/chest_xray/chest_xray
```

---

## Step 7 — Model Train করো

```bash
cd ml_pipeline
source ../backend/venv/bin/activate

# ML deps install
pip install -r requirements.txt

# Train (CPU: ~1-2 hr for 10 epochs | GPU: ~15 min)
DATA_ROOT="$(pwd)/data/chest_xray" \
SAVE_DIR="$(pwd)/models" \
python3 run_pipeline.py --skip-eda --skip-xai --epochs 10

deactivate
cd ..
```

Training arguments:
```
--epochs 5      Quick test (~30 min CPU)
--epochs 10     Better accuracy (~1-2 hr CPU)
--epochs 25     Full training (~4-5 hr CPU)
--skip-eda      Skip dataset statistics (faster)
--skip-xai      Skip XAI during training (faster)
--skip-train    Skip training, just evaluate
--eval-only     Load saved model and evaluate only
```

---

## Step 8 — Model Weights Backend-এ Copy করো

```bash
mkdir -p backend/app/ml/weights
cp ml_pipeline/models/best_model.pth backend/app/ml/weights/

# Verify
ls -lh backend/app/ml/weights/
# best_model.pth  ~28MB দেখালে OK
```

---

## Step 9 — App চালাও

```bash
./run.sh
```

অথবা আলাদা terminal-এ:
```bash
# Terminal 1 — Backend
./run.sh backend

# Terminal 2 — Frontend
./run.sh frontend
```

---

## Step 10 — Test করো

Browser-এ যাও:

| URL | কী দেখবে |
|-----|---------|
| http://localhost:8000/health | `{"status":"ok","model_loaded":true,...}` |
| http://localhost:8000/docs | Swagger API UI |
| http://localhost:5173 | React frontend |

Image upload করো → prediction + Grad-CAM + LIME দেখবে।

DB-তে data গেছে কিনা check করো:
```bash
mysql -u root -proot pneumonia_xai -e "SELECT * FROM predictions ORDER BY created_at DESC LIMIT 5;"
```

---

## Shortcut — সব একসাথে (Step 4-8)

```bash
./setup_data.sh
```

Script নিজেই করবে:
- Kaggle token check
- Dataset download
- ML deps install
- Model train
- Weights deploy to backend
- Backend DB deps install

---

## Environment Variables (Optional Override)

`backend/.env` file বানাও:
```env
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=root
DB_NAME=pneumonia_xai

MODEL_NAME=densenet121
DEVICE=cpu
DEBUG=false
```

`frontend/.env` file:
```env
VITE_API_URL=http://localhost:8000
```

---

## GPU Support (NVIDIA)

```bash
# CUDA আছে কিনা check
nvidia-smi

# backend/.env এ add করো
DEVICE=cuda

# GPU version torch install (CUDA 11.8)
source backend/venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
deactivate
```

Training-এও GPU use করবে (10x faster):
```bash
DATA_ROOT="$(pwd)/data/chest_xray" SAVE_DIR="$(pwd)/models" \
python3 run_pipeline.py --skip-eda --skip-xai --epochs 25
```

---

## Troubleshoot

| Error | Fix |
|-------|-----|
| `Can't connect to MySQL server` | `sudo systemctl start mysql` |
| `Access denied for user 'root'` | MySQL root password reset করো (Step 1 দেখো) |
| `ModuleNotFoundError` | `source backend/venv/bin/activate` করেছ? |
| `Model not found / weights empty` | Step 8 করেছ? `ls backend/app/ml/weights/` |
| `kaggle: command not found` | venv activate করে `pip install kaggle` |
| `401 Unauthorized (Kaggle)` | Token expire হয়েছে — নতুন token নাও |
| `404 Dataset not found` | Dataset name exact: `paultimothymooney/chest-xray-pneumonia` |
| `CORS error (frontend)` | Backend চলছে? `http://localhost:8000/health` check করো |
| `port already in use` | `kill $(lsof -ti:8000)` অথবা `kill $(lsof -ti:5173)` |

---

## File Structure Quick Reference

```
pneumonia_xai_fullstack/
├── run.sh                      ← App start করার script
├── setup_data.sh               ← Data download + train (one command)
├── backend/
│   ├── venv/                   ← Python virtual env (git ignored)
│   ├── requirements.txt        ← Python deps
│   ├── main.py                 ← FastAPI entry point
│   └── app/
│       ├── core/config.py      ← All settings (DB, model, image)
│       ├── db/models.py        ← MySQL predictions table
│       ├── api/predict.py      ← POST /predict
│       └── ml/weights/         ← best_model.pth রাখার জায়গা
├── frontend/
│   ├── node_modules/           ← JS deps (git ignored)
│   └── src/
│       ├── pages/Home.jsx      ← Upload + result page
│       ├── pages/Explain.jsx   ← Grad-CAM + LIME page
│       └── utils/api.js        ← Axios API client
├── ml_pipeline/
│   ├── data/chest_xray/        ← Dataset (git ignored)
│   ├── models/                 ← Trained weights (git ignored)
│   └── run_pipeline.py         ← Train entry point
└── docs/
    ├── FULL_SETUP.md           ← এই file
    ├── PROJECT.md              ← Architecture + Claude reference
    ├── API.md                  ← API endpoints
    └── DATASET_SETUP.md        ← Kaggle setup detail
```

---

## Claude-এর সাথে কাজ করার সময়

নতুন session শুরুতে বলো:

> *"এই file দেখো: https://github.com/arafattech/pneumonia-xai-fullstack/blob/main/docs/PROJECT.md — তারপর [কাজ] করো"*

Full setup-এর জন্য:

> *"FULL_SETUP.md দেখো: https://github.com/arafattech/pneumonia-xai-fullstack/blob/main/docs/FULL_SETUP.md"*
