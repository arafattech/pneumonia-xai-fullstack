# =============================================================
# app/utils/image_utils.py  —  Image loading & preprocessing
# =============================================================

import io
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from app.core.config import settings


# ── Eval transform (no augmentation) ─────────────────────────
_eval_transform = transforms.Compose([
    transforms.Resize((settings.IMG_SIZE, settings.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=settings.IMAGENET_MEAN,
                         std=settings.IMAGENET_STD),
])

# ── Resize only (for LIME which needs raw [0,1] values) ───────
_resize_only = transforms.Compose([
    transforms.Resize((settings.IMG_SIZE, settings.IMG_SIZE)),
])


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    """Convert raw bytes → RGB PIL Image."""
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """PIL Image → normalised tensor shape (1, 3, H, W)."""
    return _eval_transform(pil_img).unsqueeze(0)


def pil_to_numpy(pil_img: Image.Image) -> np.ndarray:
    """PIL Image → float32 numpy [0,1], shape (H, W, 3)."""
    resized = _resize_only(pil_img)
    return np.array(resized).astype(np.float32) / 255.0


def validate_image(content_type: str, file_size: int) -> None:
    """
    Raise ValueError if content type or file size is invalid.
    """
    if content_type not in settings.ALLOWED_TYPES:
        raise ValueError(
            f"Unsupported file type: {content_type}. "
            f"Allowed: {settings.ALLOWED_TYPES}"
        )
    if file_size > settings.MAX_FILE_SIZE:
        raise ValueError(
            f"File too large: {file_size / 1024 / 1024:.1f} MB. "
            f"Max: {settings.MAX_FILE_SIZE / 1024 / 1024:.0f} MB"
        )
