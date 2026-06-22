#!/usr/bin/env python3
# =============================================================
# ml_pipeline/run_pipeline.py
# Single entry point — runs the entire training pipeline
#
# Usage (Kaggle Notebook):
#   !python run_pipeline.py
#
# Or step by step:
#   !python run_pipeline.py --skip-eda
#   !python run_pipeline.py --skip-train --eval-only
# =============================================================

import sys
import os
import argparse
import random
import numpy as np
import torch

# ── Add src to path ──────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import config
from eda     import run_eda
from dataset import get_datasets, get_dataloaders
from model   import build_model, load_model
from train   import train
from evaluate import run_evaluation
from xai     import run_xai


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def install_packages():
    """Install optional XAI packages if missing."""
    import subprocess
    for pkg, imp in [('lime','lime'), ('shap','shap')]:
        try:
            __import__(imp)
        except ImportError:
            print(f'Installing {pkg}...')
            subprocess.check_call([sys.executable,'-m','pip','install',pkg,'-q'])


def parse_args():
    p = argparse.ArgumentParser(description='Pneumonia XAI Pipeline')
    p.add_argument('--skip-eda',   action='store_true', help='Skip EDA')
    p.add_argument('--skip-train', action='store_true', help='Skip training')
    p.add_argument('--eval-only',  action='store_true', help='Only evaluate')
    p.add_argument('--skip-xai',   action='store_true', help='Skip XAI')
    p.add_argument('--model',      default=config.MODEL_NAME,
                   help='densenet121 | resnet50 | efficientnet_b0')
    p.add_argument('--epochs',     type=int, default=config.NUM_EPOCHS)
    return p.parse_args()


def main():
    args = parse_args()

    # Apply CLI overrides
    config.MODEL_NAME  = args.model
    config.NUM_EPOCHS  = args.epochs

    print('\n' + '='*60)
    print('  🫁  PNEUMONIA XAI — FULL PIPELINE')
    print('='*60)
    config.print_config()

    # Setup
    set_seed()
    config.create_dirs()
    install_packages()

    # ── Step 1: EDA ──────────────────────────────────────────
    if not args.skip_eda and not args.eval_only:
        run_eda()

    # ── Step 2: Datasets & DataLoaders ───────────────────────
    print('\n📦  Loading datasets...')
    datasets    = get_datasets()
    dataloaders = get_dataloaders(datasets)

    # ── Step 3: Build / Load Model ───────────────────────────
    if args.eval_only and os.path.exists(config.BEST_MODEL_PATH):
        print(f'\n🔄  Loading saved model...')
        model, target_layer = load_model(config.BEST_MODEL_PATH)
    else:
        model, target_layer = build_model()
        model = model.to(config.DEVICE)

    # ── Step 4: Train ────────────────────────────────────────
    if not args.skip_train and not args.eval_only:
        history = train(model, dataloaders)

    # ── Step 5: Evaluate ─────────────────────────────────────
    metrics = run_evaluation(model, dataloaders)

    # ── Step 6: XAI ──────────────────────────────────────────
    if not args.skip_xai:
        run_xai(model, target_layer, datasets)

    # ── Final summary ────────────────────────────────────────
    print('\n' + '='*60)
    print('  🎉  PIPELINE COMPLETE')
    print('='*60)
    print(f'  Accuracy  : {metrics["accuracy"]:.2%}')
    print(f'  F1-Score  : {metrics["f1"]:.4f}')
    print(f'  ROC-AUC   : {metrics["roc_auc"]:.4f}')
    print(f'  Outputs   : {config.OUTPUT_DIR}')
    print('='*60)


if __name__ == '__main__':
    main()
