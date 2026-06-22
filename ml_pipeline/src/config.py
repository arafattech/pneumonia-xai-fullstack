# =============================================================
# ml_pipeline/src/config.py
# =============================================================

import os
import torch

# ── Dataset ───────────────────────────────────────────────────
import pathlib as _pathlib
_ML_ROOT  = str(_pathlib.Path(__file__).resolve().parents[1])
DATA_ROOT   = os.environ.get(
    'DATA_ROOT',
    os.path.join(_ML_ROOT, 'data', 'chest_xray')
)
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
NUM_CLASSES = 2
SPLITS      = ['train', 'val', 'test']

# ── Model ─────────────────────────────────────────────────────
MODEL_NAME = 'densenet121'   # 'densenet121' | 'resnet50' | 'efficientnet_b0'
PRETRAINED = True

# ── Training ──────────────────────────────────────────────────
IMG_SIZE      = 224
BATCH_SIZE    = 32
NUM_EPOCHS    = 5       # Quick test=5  |  Full=25
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 2
SEED          = 42

# ── ImageNet stats ────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── Output paths ──────────────────────────────────────────────
SAVE_DIR        = os.environ.get('SAVE_DIR', os.path.join(_ML_ROOT, 'models'))
BEST_MODEL_PATH = os.path.join(SAVE_DIR, 'best_model.pth')
LOG_PATH        = os.path.join(SAVE_DIR, 'logs', 'training_log.csv')
OUTPUT_DIR      = os.path.join(SAVE_DIR, 'outputs')
CKPT_DIR        = os.path.join(SAVE_DIR, 'checkpoints')

# ── Device ────────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_dirs():
    for d in [OUTPUT_DIR, CKPT_DIR, os.path.dirname(LOG_PATH)]:
        os.makedirs(d, exist_ok=True)


def print_config():
    print('=' * 52)
    print('  CONFIGURATION')
    print('=' * 52)
    print(f'  Model       : {MODEL_NAME}')
    print(f'  Device      : {DEVICE}')
    print(f'  Epochs      : {NUM_EPOCHS}')
    print(f'  Batch size  : {BATCH_SIZE}')
    print(f'  Image size  : {IMG_SIZE}×{IMG_SIZE}')
    print(f'  LR          : {LEARNING_RATE}')
    print(f'  Data root   : {DATA_ROOT}')
    print('=' * 52)
