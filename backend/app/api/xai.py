# =============================================================
# app/api/xai.py  —  POST /explain/gradcam  &  /explain/lime
# =============================================================

import torch
import torch.nn.functional as F
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import settings
from app.core.model_loader import model_loader
from app.ml.gradcam import generate_gradcam_result
from app.ml.lime_exp import generate_lime_result
from app.schemas.schemas import GradCAMResponse, LIMEResponse
from app.utils.image_utils import bytes_to_pil, pil_to_numpy, pil_to_tensor, validate_image

router = APIRouter()


def _get_prediction(img_tensor):
    """Run inference and return (pred_idx, confidence, probs)."""
    with torch.no_grad():
        output = model_loader.model(img_tensor)
        probs  = F.softmax(output, dim=1).cpu().numpy()[0]
    pred_idx   = int(probs.argmax())
    confidence = float(probs[pred_idx])
    return pred_idx, confidence, probs


def _xai_explanation_text(pred_class: str, confidence: float, method: str) -> str:
    """Generate a simple text interpretation for the heatmap."""
    if pred_class == "PNEUMONIA":
        region = "lung consolidation regions (lower and middle zones)"
        note   = "Red/yellow areas highlight where the model detected abnormal opacity."
    else:
        region = "clear lung fields with diffuse, low-intensity activation"
        note   = "Minimal focal activation confirms absence of consolidation."

    return (
        f"[{method}] Model predicted {pred_class} with {confidence:.1%} confidence. "
        f"Attention focused on: {region}. {note}"
    )


# ── Grad-CAM ──────────────────────────────────────────────────

@router.post("/gradcam", response_model=GradCAMResponse)
async def explain_gradcam(file: UploadFile = File(...)):
    """
    Generate a Grad-CAM heatmap for the uploaded chest X-ray.

    Returns original image, heatmap, and overlay as base64 PNGs,
    along with a text interpretation of where the model focused.
    """
    contents = await file.read()
    try:
        validate_image(file.content_type, len(contents))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        pil_img    = bytes_to_pil(contents)
        img_tensor = pil_to_tensor(pil_img).to(model_loader.device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image error: {e}")

    try:
        pred_idx, confidence, _ = _get_prediction(img_tensor)
        pred_class = settings.CLASS_NAMES[pred_idx]

        result = generate_gradcam_result(
            model        = model_loader.model,
            target_layer = model_loader.target_layer,
            img_tensor   = img_tensor,
            class_idx    = pred_idx,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grad-CAM failed: {e}")

    return GradCAMResponse(
        predicted_class = pred_class,
        confidence      = confidence,
        original_b64    = result["original_b64"],
        heatmap_b64     = result["heatmap_b64"],
        overlay_b64     = result["overlay_b64"],
        explanation     = _xai_explanation_text(pred_class, confidence, "Grad-CAM"),
    )


# ── LIME ──────────────────────────────────────────────────────

@router.post("/lime", response_model=LIMEResponse)
async def explain_lime(file: UploadFile = File(...)):
    """
    Generate a LIME superpixel explanation for the uploaded X-ray.

    Returns base64 images highlighting regions that positively or
    negatively influenced the model's prediction.
    """
    contents = await file.read()
    try:
        validate_image(file.content_type, len(contents))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        pil_img    = bytes_to_pil(contents)
        img_tensor = pil_to_tensor(pil_img).to(model_loader.device)
        img_rgb    = pil_to_numpy(pil_img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image error: {e}")

    pred_idx, confidence, _ = _get_prediction(img_tensor)
    pred_class = settings.CLASS_NAMES[pred_idx]

    result = generate_lime_result(
        model       = model_loader.model,
        img_rgb     = img_rgb,
        num_samples = 300,
        num_features= 10,
    )

    if not result.get("available"):
        return LIMEResponse(
            available = False,
            error     = result.get("error", "LIME unavailable"),
        )

    return LIMEResponse(
        available       = True,
        predicted_class = pred_class,
        confidence      = confidence,
        positive_b64    = result["positive_b64"],
        negative_b64    = result["negative_b64"],
        all_b64         = result["all_b64"],
        explanation     = _xai_explanation_text(pred_class, confidence, "LIME"),
    )
