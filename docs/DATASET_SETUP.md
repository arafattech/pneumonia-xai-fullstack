# Dataset Setup Guide

Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

---

## Step 1: Kaggle Account

1. [kaggle.com](https://www.kaggle.com) এ account বানাও (না থাকলে)
2. Login করো

---

## Step 2: Kaggle API Key বানাও

1. `https://www.kaggle.com/settings/account` যাও
2. **API** section খোঁজো
3. **"Create New Token"** click করো
4. `kaggle.json` file download হবে

---

## Step 3: kaggle.json রাখো

```bash
mkdir -p ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Verify:
```bash
cat ~/.kaggle/kaggle.json
# Output হবে: {"username":"your_username","key":"xxxxx"}
```

---

## Step 4: Kaggle CLI install করো

```bash
cd ~/Downloads/pneumonia_xai_fullstack/backend
source venv/bin/activate
pip install kaggle
kaggle --version
deactivate
```

---

## Step 5: Dataset download করো

```bash
mkdir -p ~/Downloads/pneumonia_xai_fullstack/ml_pipeline/data
cd ~/Downloads/pneumonia_xai_fullstack/ml_pipeline/data

source ~/Downloads/pneumonia_xai_fullstack/backend/venv/bin/activate
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia --unzip
deactivate
```

Download হলে এই structure হবে:

```
ml_pipeline/data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/       (~1341 images)
    │   └── PNEUMONIA/    (~3875 images)
    ├── val/
    │   ├── NORMAL/       (8 images)
    │   └── PNEUMONIA/    (8 images)
    └── test/
        ├── NORMAL/       (~234 images)
        └── PNEUMONIA/    (~390 images)
```

---

## Step 6: Folder structure fix করো (দরকার হলে)

Kaggle sometimes nested folder বানায়:

```bash
# Check করো
ls ~/Downloads/pneumonia_xai_fullstack/ml_pipeline/data/

# যদি chest_xray/chest_xray/ double nested থাকে:
mv ~/Downloads/pneumonia_xai_fullstack/ml_pipeline/data/chest_xray/chest_xray/* \
   ~/Downloads/pneumonia_xai_fullstack/ml_pipeline/data/chest_xray/
rmdir ~/Downloads/pneumonia_xai_fullstack/ml_pipeline/data/chest_xray/chest_xray
```

---

## Step 7: Model train করো

```bash
cd ~/Downloads/pneumonia_xai_fullstack/ml_pipeline
source ../backend/venv/bin/activate

# Quick test (5 epochs, ~30 min CPU)
DATA_ROOT="$(pwd)/data/chest_xray" \
SAVE_DIR="$(pwd)/models" \
python3 run_pipeline.py --skip-eda --epochs 5

deactivate
```

Train হলে model save হবে: `ml_pipeline/models/best_model.pth`

---

## Step 8: Weights backend-এ copy করো

```bash
mkdir -p ~/Downloads/pneumonia_xai_fullstack/backend/app/ml/weights
cp ~/Downloads/pneumonia_xai_fullstack/ml_pipeline/models/best_model.pth \
   ~/Downloads/pneumonia_xai_fullstack/backend/app/ml/weights/
```

---

## Step 9: App চালাও

```bash
cd ~/Downloads/pneumonia_xai_fullstack
./run.sh
```

- API: http://localhost:8000/docs
- Frontend: http://localhost:5173

---

## Shortcut: সব একসাথে

```bash
cd ~/Downloads/pneumonia_xai_fullstack
./setup_data.sh
```

Script নিজেই step 4–8 করবে।

---

## Troubleshoot

| Error | Fix |
|-------|-----|
| `401 - Unauthorized` | `~/.kaggle/kaggle.json` সঠিক আছে? `chmod 600` দিয়েছ? |
| `404 - Not Found` | Dataset name exact হতে হবে: `paultimothymooney/chest-xray-pneumonia` |
| `No module named kaggle` | `source venv/bin/activate` করে `pip install kaggle` |
| `FileNotFoundError: chest_xray` | Step 6 দেখো — nested folder fix করো |
| Model load error | `best_model.pth` সঠিক path-এ আছে? Step 8 দেখো |
