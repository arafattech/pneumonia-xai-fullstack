# =============================================================
# app/schemas/schemas.py  —  Pydantic request / response models
# =============================================================

from pydantic import BaseModel, Field
from typing import Optional


# ── Prediction ────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    predicted_class:   str           = Field(..., example="PNEUMONIA")
    predicted_index:   int           = Field(..., example=1)
    confidence:        float         = Field(..., example=0.9732)
    probabilities:     dict[str, float]
    model_name:        str           = Field(..., example="densenet121")
    image_size:        tuple[int, int]

    class Config:
        json_schema_extra = {
            "example": {
                "predicted_class": "PNEUMONIA",
                "predicted_index": 1,
                "confidence": 0.9732,
                "probabilities": {"NORMAL": 0.0268, "PNEUMONIA": 0.9732},
                "model_name": "densenet121",
                "image_size": [224, 224],
            }
        }


# ── Grad-CAM ──────────────────────────────────────────────────

class GradCAMResponse(BaseModel):
    predicted_class:  str
    confidence:       float
    original_b64:     str    = Field(..., description="Base64 PNG of original image")
    heatmap_b64:      str    = Field(..., description="Base64 PNG of Grad-CAM heatmap")
    overlay_b64:      str    = Field(..., description="Base64 PNG of overlay")
    explanation:      str    = Field(..., description="Text interpretation of heatmap")


# ── LIME ──────────────────────────────────────────────────────

class LIMEResponse(BaseModel):
    available:        bool
    predicted_class:  Optional[str]   = None
    confidence:       Optional[float] = None
    positive_b64:     Optional[str]   = None
    negative_b64:     Optional[str]   = None
    all_b64:          Optional[str]   = None
    explanation:      Optional[str]   = None
    error:            Optional[str]   = None


# ── Health ────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:      str
    model_loaded: bool
    model_name:  str
    device:      str
    version:     str
