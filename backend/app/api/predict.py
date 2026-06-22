# =============================================================
# app/api/predict.py  —  POST /predict
# =============================================================

import torch
import torch.nn.functional as F
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.model_loader import model_loader
from app.db.database import get_db
from app.db.models import Prediction
from app.schemas.schemas import PredictionResponse
from app.utils.image_utils import bytes_to_pil, pil_to_tensor, validate_image

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # ── Validate ──────────────────────────────────────────────
    contents = await file.read()
    try:
        validate_image(file.content_type, len(contents))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # ── Preprocess ────────────────────────────────────────────
    try:
        pil_img    = bytes_to_pil(contents)
        img_tensor = pil_to_tensor(pil_img).to(model_loader.device)
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail=f"Image processing failed: {e}")

    # ── Inference ─────────────────────────────────────────────
    try:
        with torch.no_grad():
            output = model_loader.model(img_tensor)
            probs  = F.softmax(output, dim=1).cpu().numpy()[0]
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Model inference failed: {e}")

    pred_idx   = int(probs.argmax())
    pred_class = settings.CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])
    prob_dict  = {
        name: float(p)
        for name, p in zip(settings.CLASS_NAMES, probs)
    }

    # ── Save to DB ────────────────────────────────────────────
    try:
        db.add(Prediction(
            filename        = file.filename,
            predicted_class = pred_class,
            predicted_index = pred_idx,
            confidence      = confidence,
            prob_normal     = float(probs[0]),
            prob_pneumonia  = float(probs[1]),
            model_name      = settings.MODEL_NAME,
        ))
        db.commit()
    except Exception:
        db.rollback()

    return PredictionResponse(
        predicted_class = pred_class,
        predicted_index = pred_idx,
        confidence      = confidence,
        probabilities   = prob_dict,
        model_name      = settings.MODEL_NAME,
        image_size      = (settings.IMG_SIZE, settings.IMG_SIZE),
    )
