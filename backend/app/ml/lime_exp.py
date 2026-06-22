# =============================================================
# app/ml/lime_exp.py  —  LIME image explanation
# Ref: Ribeiro et al. 2016  https://arxiv.org/abs/1602.04938
# =============================================================

import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from app.core.config import settings
from app.ml.gradcam import numpy_to_base64

try:
    from lime import lime_image
    from skimage.segmentation import mark_boundaries
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


def _build_predict_fn(model):
    """
    Build a predict function compatible with LIME's API.
    Input:  numpy [N, H, W, 3] float32 in [0, 1]
    Output: numpy [N, num_classes] softmax probabilities
    """
    norm = transforms.Normalize(settings.IMAGENET_MEAN, settings.IMAGENET_STD)
    to_t = transforms.ToTensor()

    def predict_fn(images_np: np.ndarray) -> np.ndarray:
        batch = []
        for img in images_np:
            pil = Image.fromarray((img * 255).astype(np.uint8)).resize(
                (settings.IMG_SIZE, settings.IMG_SIZE))
            batch.append(norm(to_t(pil)))
        with torch.no_grad():
            out  = model(torch.stack(batch).to(settings.DEVICE))
            prob = F.softmax(out, dim=1).cpu().numpy()
        return prob

    return predict_fn


def generate_lime_result(
    model,
    img_rgb: np.ndarray,       # float32 [0,1] HWC
    num_samples: int = 300,
    num_features: int = 10,
    seed: int = 42,
) -> dict:
    """
    Run LIME and return explanation images as base64 strings.

    Returns dict with:
      positive_b64   — regions supporting prediction
      negative_b64   — regions contradicting prediction
      all_b64        — all significant regions
      available      — bool: was LIME library found
    """
    if not LIME_AVAILABLE:
        return {
            "available": False,
            "error": "LIME not installed. Run: pip install lime",
        }

    predict_fn = _build_predict_fn(model)
    explainer  = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image          = img_rgb,
        classifier_fn  = predict_fn,
        top_labels     = 2,
        hide_color     = 0,
        num_samples    = num_samples,
        random_seed    = seed,
    )

    pred_label = explanation.top_labels[0]

    def get_img(positive_only, negative_only=False):
        temp, mask = explanation.get_image_and_mask(
            label          = pred_label,
            positive_only  = positive_only,
            negative_only  = negative_only,
            num_features   = num_features,
            hide_rest      = False,
        )
        result = mark_boundaries(temp, mask)
        return numpy_to_base64(np.clip(result, 0, 1))

    return {
        "available":    True,
        "positive_b64": get_img(positive_only=True),
        "negative_b64": get_img(positive_only=False, negative_only=True),
        "all_b64":      numpy_to_base64(img_rgb),
        "pred_label":   int(pred_label),
    }
