# =============================================================
# app/ml/gradcam.py  —  Grad-CAM heatmap generator
# Ref: Selvaraju et al. 2017  https://arxiv.org/abs/1610.02391
# =============================================================

import base64
import io

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from app.core.config import settings


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Works with any CNN that has a registered target layer.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.activations: torch.Tensor | None = None
        self.gradients:   torch.Tensor | None = None

        # Register hooks
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inp, output):
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,  # (1, 3, H, W)
        class_idx: int | None = None,
    ) -> np.ndarray:
        """
        Returns normalised float32 CAM in [0, 1], shape (H, W).
        """
        self.model.eval()
        input_tensor = input_tensor.to(settings.DEVICE).requires_grad_(True)

        output    = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, class_idx].backward()

        # Global average pool gradients over spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * self.activations).sum(dim=1).squeeze())
        cam     = cam.cpu().detach().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def overlay_on_image(
        self,
        img_rgb: np.ndarray,   # float32 [0,1], shape (H,W,3)
        cam:     np.ndarray,   # float32 [0,1], shape (h,w)
        alpha:   float = 0.45,
    ) -> np.ndarray:
        """Resize cam and overlay as jet colormap on the image."""
        cam_r  = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))
        hmap   = cv2.applyColorMap(np.uint8(255 * cam_r), cv2.COLORMAP_JET)
        hmap   = cv2.cvtColor(hmap, cv2.COLOR_BGR2RGB) / 255.0
        result = np.clip((1 - alpha) * img_rgb + alpha * hmap, 0, 1)
        return result


# ── Helpers ───────────────────────────────────────────────────

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalisation → float32 HWC [0,1]."""
    mean = torch.tensor(settings.IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(settings.IMAGENET_STD).view(3, 1, 1)
    img  = (tensor.cpu().detach() * std + mean).permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1)


def numpy_to_base64(img_float: np.ndarray) -> str:
    """Convert float32 [0,1] numpy image to base64-encoded PNG string."""
    img_uint8 = (img_float * 255).astype(np.uint8)
    pil_img   = Image.fromarray(img_uint8)
    buf       = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def generate_gradcam_result(
    model:        nn.Module,
    target_layer: nn.Module,
    img_tensor:   torch.Tensor,   # (1, 3, 224, 224)
    class_idx:    int | None = None,
) -> dict:
    """
    Full Grad-CAM pipeline.

    Returns dict with:
      original_b64   — original image as base64 PNG
      heatmap_b64    — raw jet heatmap as base64 PNG
      overlay_b64    — overlay image as base64 PNG
      cam_array      — raw CAM values (list of lists)
    """
    gc      = GradCAM(model, target_layer)
    cam     = gc.generate_cam(img_tensor, class_idx=class_idx)
    img_rgb = denormalize(img_tensor.squeeze(0))
    overlay = gc.overlay_on_image(img_rgb, cam)

    # Raw heatmap (jet colormap)
    cam_r   = cv2.resize(cam, (img_rgb.shape[1], img_rgb.shape[0]))
    hmap    = cv2.applyColorMap(np.uint8(255 * cam_r), cv2.COLORMAP_JET)
    hmap    = cv2.cvtColor(hmap, cv2.COLOR_BGR2RGB) / 255.0

    return {
        "original_b64": numpy_to_base64(img_rgb),
        "heatmap_b64":  numpy_to_base64(hmap),
        "overlay_b64":  numpy_to_base64(overlay),
        "cam_array":    cam.tolist(),
    }
