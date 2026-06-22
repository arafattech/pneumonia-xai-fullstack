# =============================================================
# ml_pipeline/src/xai.py  —  Grad-CAM, LIME, SHAP
# =============================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import config

try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

COLORS = {'NORMAL': '#27ae60', 'PNEUMONIA': '#e74c3c'}


# ── Helpers ───────────────────────────────────────────────────

def denormalize(tensor):
    mean = torch.tensor(config.IMAGENET_MEAN).view(3,1,1)
    std  = torch.tensor(config.IMAGENET_STD).view(3,1,1)
    img  = (tensor.cpu() * std + mean).permute(1,2,0).numpy()
    return np.clip(img, 0, 1)


def overlay_heatmap(img_rgb, cam, alpha=0.45):
    cam_r  = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))
    hmap   = cv2.applyColorMap(np.uint8(255 * cam_r), cv2.COLORMAP_JET)
    hmap   = cv2.cvtColor(hmap, cv2.COLOR_BGR2RGB) / 255.0
    return np.clip((1-alpha)*img_rgb + alpha*hmap, 0, 1), cam_r


# ── Grad-CAM ──────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.activations = None
        self.gradients   = None
        target_layer.register_forward_hook(
            lambda m,i,o: setattr(self,'activations', o.detach()))
        target_layer.register_full_backward_hook(
            lambda m,gi,go: setattr(self,'gradients', go[0].detach()))

    def __call__(self, img_tensor, class_idx=None):
        self.model.eval()
        x   = img_tensor.unsqueeze(0).to(config.DEVICE).requires_grad_(True)
        out = self.model(x)
        if class_idx is None:
            class_idx = out.argmax(1).item()
        self.model.zero_grad()
        out[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        cam     = F.relu((weights * self.activations).sum(1).squeeze())
        cam     = cam.cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


@torch.no_grad()
def collect_samples(model, dataset, device, n_correct=4, n_wrong=2):
    model.eval()
    correct, wrong = [], []
    for idx in range(len(dataset)):
        img, lbl = dataset[idx]
        pred = model(img.unsqueeze(0).to(device)).argmax(1).item()
        if pred == lbl and len(correct) < n_correct:
            correct.append(idx)
        elif pred != lbl and len(wrong) < n_wrong:
            wrong.append(idx)
        if len(correct) >= n_correct and len(wrong) >= n_wrong:
            break
    return correct, wrong


def plot_gradcam(model, gradcam_fn, dataset, indices, title, save_path):
    n   = len(indices)
    if n == 0:
        print('  No samples to plot.')
        return
    fig, axes = plt.subplots(n, 3, figsize=(13, 4.5*n))
    if n == 1: axes = [axes]
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
    model.eval()
    for row, idx in enumerate(indices):
        img_t, true_lbl = dataset[idx]
        with torch.no_grad():
            prob = F.softmax(
                model(img_t.unsqueeze(0).to(config.DEVICE)), dim=1
            ).cpu().numpy()[0]
        pred_lbl   = prob.argmax()
        confidence = prob[pred_lbl]
        cam        = gradcam_fn(img_t, class_idx=int(pred_lbl))
        img_rgb    = denormalize(img_t)
        overlay, cam_r = overlay_heatmap(img_rgb, cam)
        ok    = pred_lbl == true_lbl
        color = '#27ae60' if ok else '#e74c3c'
        mark  = '✓' if ok else '✗'
        for col, (data, ttl, cm_) in enumerate([
            (img_rgb,  f'Original\nTrue: {config.CLASS_NAMES[true_lbl]}', None),
            (cam_r,    'Grad-CAM\nHeatmap', 'jet'),
            (overlay,  f'{mark} Pred: {config.CLASS_NAMES[pred_lbl]}\nConf: {confidence:.2%}', None),
        ]):
            ax = axes[row][col]
            ax.imshow(data, cmap=cm_)
            ax.set_title(ttl, fontsize=10,
                         color=color if col==2 else 'black',
                         fontweight='bold' if col==2 else 'normal')
            ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'✅  {save_path}')


# ── LIME ──────────────────────────────────────────────────────

def run_lime(model, dataset, indices, save_dir, num_samples=400):
    if not LIME_AVAILABLE:
        print('⚠️  LIME not installed. pip install lime')
        return

    norm = transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD)
    to_t = transforms.ToTensor()

    def predict_fn(imgs_np):
        batch = []
        for img in imgs_np:
            pil = Image.fromarray((img*255).astype(np.uint8)).resize(
                (config.IMG_SIZE, config.IMG_SIZE))
            batch.append(norm(to_t(pil)))
        with torch.no_grad():
            out = model(torch.stack(batch).to(config.DEVICE))
        return F.softmax(out, dim=1).cpu().numpy()

    for i, idx in enumerate(indices[:2]):
        img_t, true_lbl = dataset[idx]
        img_rgb = denormalize(img_t)
        exp = lime_image.LimeImageExplainer().explain_instance(
            image=img_rgb, classifier_fn=predict_fn,
            top_labels=2, hide_color=0,
            num_samples=num_samples, random_seed=config.SEED)
        pred_lbl = exp.top_labels[0]
        with torch.no_grad():
            prob = F.softmax(
                model(img_t.unsqueeze(0).to(config.DEVICE)), dim=1
            ).cpu().numpy()[0]
        temp, mask = exp.get_image_and_mask(
            label=pred_lbl, positive_only=True,
            num_features=10, hide_rest=False)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ok = pred_lbl == true_lbl
        fig.suptitle(
            f'LIME  |  True: {config.CLASS_NAMES[true_lbl]}  |  '
            f'{"✓" if ok else "✗"} Pred: {config.CLASS_NAMES[pred_lbl]} '
            f'({prob[pred_lbl]:.2%})',
            fontsize=12, fontweight='bold')
        ax1.imshow(img_rgb); ax1.set_title('Original'); ax1.axis('off')
        ax2.imshow(mark_boundaries(temp, mask))
        ax2.set_title('LIME: Supporting regions'); ax2.axis('off')
        plt.tight_layout()
        path = os.path.join(save_dir, f'lime_sample_{i}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f'✅  {path}')


# ── SHAP ──────────────────────────────────────────────────────

def run_shap(model, dataset, n_bg=20):
    if not SHAP_AVAILABLE:
        print('⚠️  SHAP not installed. pip install shap')
        return
    try:
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=n_bg, shuffle=True)
        bg, _  = next(iter(loader))
        bg     = bg.to(config.DEVICE)
        exp    = shap.GradientExplainer(model, bg)
        sample_imgs = torch.stack([dataset[i][0] for i in range(2)]).to(config.DEVICE)
        vals = exp.shap_values(sample_imgs, ranked_outputs=1)
        imgs_np = sample_imgs.cpu().numpy().transpose(0,2,3,1)
        shap.image_plot(vals[0], imgs_np)
        path = os.path.join(config.OUTPUT_DIR, 'shap.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f'✅  {path}')
    except Exception as e:
        print(f'⚠️  SHAP failed: {e}')


# ── Full XAI run ──────────────────────────────────────────────

def run_xai(model, target_layer, datasets):
    print('\n🔥  Running XAI Analysis...')
    gradcam_fn = GradCAM(model, target_layer)
    test_ds    = datasets['test']

    print('  Collecting correct/wrong predictions...')
    correct_idx, wrong_idx = collect_samples(
        model, test_ds, config.DEVICE, n_correct=4, n_wrong=2)
    print(f'  Correct: {len(correct_idx)}  |  Wrong: {len(wrong_idx)}')

    plot_gradcam(model, gradcam_fn, test_ds, correct_idx,
                 'Grad-CAM — Correct Predictions',
                 os.path.join(config.OUTPUT_DIR, 'gradcam_correct.png'))

    if wrong_idx:
        plot_gradcam(model, gradcam_fn, test_ds, wrong_idx,
                     'Grad-CAM — Wrong Predictions (Failure Analysis)',
                     os.path.join(config.OUTPUT_DIR, 'gradcam_wrong.png'))

    run_lime(model, test_ds, correct_idx, config.OUTPUT_DIR)
    run_shap(model, test_ds)
    print('\n✅  XAI complete')
