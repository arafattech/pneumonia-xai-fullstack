# =============================================================
# ml_pipeline/src/evaluate.py  —  Test set evaluation
# =============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    classification_report
)
import config


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for imgs, labels in loader:
        imgs   = imgs.to(device)
        out    = model(imgs)
        probs  = F.softmax(out, dim=1).cpu().numpy()
        y_true.extend(labels.numpy())
        y_pred.extend(probs.argmax(axis=1))
        y_prob.extend(probs)
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


def compute_metrics(y_true, y_pred, y_prob):
    metrics = {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall':    recall_score(y_true, y_pred, average='binary'),
        'f1':        f1_score(y_true, y_pred, average='binary'),
    }
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
    except Exception:
        metrics['roc_auc'] = float('nan')
    return metrics


def print_metrics(metrics, y_true, y_pred):
    print('\n' + '='*52)
    print('  TEST SET EVALUATION')
    print('='*52)
    print(f'  Accuracy   : {metrics["accuracy"]:.4f}  ({metrics["accuracy"]:.2%})')
    print(f'  Precision  : {metrics["precision"]:.4f}')
    print(f'  Recall     : {metrics["recall"]:.4f}  ← most important')
    print(f'  F1-Score   : {metrics["f1"]:.4f}')
    print(f'  ROC-AUC    : {metrics["roc_auc"]:.4f}')
    print('='*52)
    print()
    print(classification_report(y_true, y_pred, target_names=config.CLASS_NAMES))


def plot_evaluation(y_true, y_pred, y_prob, save_path=None):
    if save_path is None:
        save_path = os.path.join(config.OUTPUT_DIR, 'evaluation.png')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Model Evaluation', fontsize=15, fontweight='bold')

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.CLASS_NAMES,
                yticklabels=config.CLASS_NAMES,
                linewidths=0.8, ax=ax1, annot_kws={'size': 15, 'weight': 'bold'})
    for i, row in enumerate([['TN','FP'],['FN','TP']]):
        for j, lbl in enumerate(row):
            ax1.text(j+0.5, i+0.75, lbl, ha='center', color='gray', fontsize=9)
    ax1.set(title='Confusion Matrix', ylabel='True', xlabel='Predicted')

    # ROC curve
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:,1])
        auc_val     = roc_auc_score(y_true, y_prob[:,1])
        ax2.plot(fpr, tpr, '#e74c3c', lw=2.5, label=f'AUC = {auc_val:.4f}')
        ax2.fill_between(fpr, tpr, alpha=0.12, color='#e74c3c')
    except Exception:
        pass
    ax2.plot([0,1],[0,1],'k--',lw=1.5,label='Random')
    ax2.set(xlim=[0,1], ylim=[0,1.02],
            xlabel='False Positive Rate', ylabel='True Positive Rate',
            title='ROC Curve')
    ax2.legend(loc='lower right', fontsize=11)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'✅  {save_path}')


def run_evaluation(model, dataloaders):
    print('\n📈  Running evaluation on test set...')
    y_true, y_pred, y_prob = get_predictions(
        model, dataloaders['test'], config.DEVICE)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics, y_true, y_pred)
    plot_evaluation(y_true, y_pred, y_prob)
    return metrics
