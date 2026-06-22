# =============================================================
# ml_pipeline/src/train.py  —  Training loop
# =============================================================

import copy, os, time
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import config


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += labels.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum = correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
        loss = criterion(out, labels)
        loss_sum += loss.item() * imgs.size(0)
        correct  += (out.argmax(1) == labels).sum().item()
        total    += labels.size(0)
    return loss_sum / total, correct / total


def train(model, dataloaders, device=config.DEVICE):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE,
                           weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)

    history      = {k: [] for k in ['train_loss','train_acc','val_loss','val_acc']}
    best_val_acc = 0.0
    best_wts     = copy.deepcopy(model.state_dict())
    log_rows     = []

    print('\n🚀  Training started')
    print('='*72)
    print(f'{"Ep":>3} | {"T-Loss":>9} | {"T-Acc":>7} | {"V-Loss":>9} | {"V-Acc":>7} | {"LR":>10} | Time')
    print('-'*72)

    for epoch in range(1, config.NUM_EPOCHS + 1):
        t0 = time.time()
        tl, ta = train_epoch(model, dataloaders['train'], criterion, optimizer, device)
        vl, va = eval_epoch( model, dataloaders['val'],   criterion,            device)
        scheduler.step()

        for k, v in zip(history, [tl, ta, vl, va]):
            history[k].append(v)

        lr  = optimizer.param_groups[0]['lr']
        tag = ''
        if va > best_val_acc:
            best_val_acc = va
            best_wts     = copy.deepcopy(model.state_dict())
            torch.save(best_wts, config.BEST_MODEL_PATH)
            ckpt = os.path.join(config.CKPT_DIR, f'checkpoint_ep{epoch:02d}.pth')
            torch.save({'epoch': epoch, 'model_state': best_wts,
                        'val_acc': va}, ckpt)
            tag = '  ◀ BEST'

        print(f'{epoch:>3} | {tl:>9.4f} | {ta:>6.2%} | {vl:>9.4f} | '
              f'{va:>6.2%} | {lr:>10.2e} | {time.time()-t0:.0f}s{tag}')
        log_rows.append({'epoch': epoch, 'train_loss': tl, 'train_acc': ta,
                         'val_loss': vl, 'val_acc': va, 'lr': lr})

    print('='*72)
    print(f'\n✅  Best Val Accuracy: {best_val_acc:.2%}')
    print(f'    Model → {config.BEST_MODEL_PATH}')

    pd.DataFrame(log_rows).to_csv(config.LOG_PATH, index=False)
    model.load_state_dict(best_wts)
    _plot_curves(history)
    return history


def _plot_curves(history):
    path = os.path.join(config.OUTPUT_DIR, 'training_curves.png')
    eps  = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Training — {config.MODEL_NAME.upper()}',
                 fontsize=14, fontweight='bold')
    ax1.plot(eps, history['train_loss'], 'b-o', lw=2, ms=5, label='Train')
    ax1.plot(eps, history['val_loss'],   'r-s', lw=2, ms=5, label='Val')
    ax1.set(xlabel='Epoch', ylabel='Loss', title='Cross-Entropy Loss')
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(eps, [a*100 for a in history['train_acc']], 'b-o', lw=2, ms=5, label='Train')
    ax2.plot(eps, [a*100 for a in history['val_acc']],   'r-s', lw=2, ms=5, label='Val')
    best = max(history['val_acc'])
    ax2.axhline(best*100, color='green', ls='--', lw=1.5, label=f'Best={best:.1%}')
    ax2.set(xlabel='Epoch', ylabel='Accuracy (%)', title='Accuracy')
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'✅  {path}')
