# =============================================================
# ml_pipeline/src/eda.py  —  Exploratory Data Analysis
# =============================================================

import os
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

import config

COLORS = {'NORMAL': '#27ae60', 'PNEUMONIA': '#e74c3c'}


def validate_dataset(root=config.DATA_ROOT):
    root    = Path(root)
    missing = []
    for split in config.SPLITS:
        for cls in config.CLASS_NAMES:
            p = root / split / cls
            if not p.exists():
                missing.append(str(p))
    if missing:
        print('❌  Missing folders:')
        for m in missing: print(f'    {m}')
        print('\n👉  https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia')
        return False
    print(f'✅  Dataset at: {root}')
    return True


def count_images(root=config.DATA_ROOT):
    root    = Path(root)
    records = []
    exts    = ['*.jpeg','*.jpg','*.png','*.JPEG','*.JPG']
    for split in config.SPLITS:
        for cls in config.CLASS_NAMES:
            folder = root / split / cls
            n = sum(len(list(folder.glob(e))) for e in exts)
            records.append({'Split': split, 'Class': cls, 'Count': n})
    return pd.DataFrame(records)


def plot_distribution(df, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Dataset Class Distribution', fontsize=15, fontweight='bold', y=1.02)
    for ax, split in zip(axes, config.SPLITS):
        sub  = df[df['Split'] == split]
        bars = ax.bar(sub['Class'], sub['Count'],
                      color=[COLORS[c] for c in sub['Class']],
                      edgecolor='black', linewidth=0.8, alpha=0.88, width=0.45)
        for bar, val in zip(bars, sub['Count']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+8,
                    f'{val:,}', ha='center', fontweight='bold', fontsize=12)
        ax.set_title(f'{split.upper()} Set', fontsize=13, fontweight='bold')
        ax.set_ylabel('Image Count')
        ax.set_ylim(0, sub['Count'].max()*1.2)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.spines[['top','right']].set_visible(False)
    patches = [mpatches.Patch(color=COLORS[c], label=c) for c in COLORS]
    fig.legend(handles=patches, loc='upper right', fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'✅  {save_path}')
    plt.show()


def plot_samples(root=config.DATA_ROOT, n=4, save_path=None):
    root = Path(root)
    fig, axes = plt.subplots(2, n, figsize=(4*n, 9))
    fig.suptitle('Sample X-Ray Images — Training Set', fontsize=16, fontweight='bold')
    for row, cls in enumerate(config.CLASS_NAMES):
        folder = root / 'train' / cls
        imgs   = list(folder.glob('*.jpeg')) + list(folder.glob('*.jpg'))
        paths  = random.sample(imgs, min(n, len(imgs)))
        for col, p in enumerate(paths):
            ax = axes[row][col]
            ax.imshow(Image.open(p).convert('RGB'), cmap='gray')
            ax.set_title(cls, color=COLORS[cls], fontweight='bold', fontsize=11)
            ax.set_xlabel(p.name[:20], fontsize=7, color='gray')
            ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'✅  {save_path}')
    plt.show()


def analyze_sizes(root=config.DATA_ROOT, n=80, save_path=None):
    root    = Path(root)
    records = []
    for cls in config.CLASS_NAMES:
        folder = root / 'train' / cls
        imgs   = list(folder.glob('*.jpeg')) + list(folder.glob('*.jpg'))
        imgs   = random.sample(imgs, min(n, len(imgs)))
        for p in imgs:
            try:
                w, h = Image.open(p).size
                records.append({'Class': cls, 'Width': w, 'Height': h})
            except: pass
    df  = pd.DataFrame(records)
    print('\nImage Size Statistics:')
    print(df.groupby('Class')[['Width','Height']].describe().round(1))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Image Size Distribution', fontsize=14, fontweight='bold')
    for ax, col in zip(axes, ['Width','Height']):
        for cls, clr in COLORS.items():
            ax.hist(df[df['Class']==cls][col], bins=30, alpha=0.6,
                    color=clr, label=cls, edgecolor='black', lw=0.4)
        ax.set_xlabel(f'{col} (px)'); ax.set_ylabel('Count')
        ax.set_title(f'{col} Distribution'); ax.legend()
        ax.grid(alpha=0.3); ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'✅  {save_path}')
    plt.show()
    return df


def run_eda():
    print('\n📊  Exploratory Data Analysis\n')
    if not validate_dataset():
        raise FileNotFoundError('Dataset missing.')
    df = count_images()
    pivot = df.pivot(index='Split', columns='Class', values='Count')
    pivot['Total'] = pivot.sum(axis=1)
    pivot['Ratio'] = (pivot['PNEUMONIA'] / pivot['NORMAL']).round(2)
    print(pivot.to_string())
    print(f"\nGrand Total: {pivot['Total'].sum():,}")
    plot_distribution(df, os.path.join(config.OUTPUT_DIR, 'class_distribution.png'))
    plot_samples(save_path=os.path.join(config.OUTPUT_DIR, 'sample_images.png'))
    analyze_sizes(save_path=os.path.join(config.OUTPUT_DIR, 'image_sizes.png'))
    print('\n✅  EDA complete\n')
