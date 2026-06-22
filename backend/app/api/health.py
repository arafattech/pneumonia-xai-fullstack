# =============================================================
# app/api/health.py  —  Health check endpoint
# =============================================================

from fastapi import APIRouter
from app.schemas.schemas import HealthResponse
from app.core.config import settings
from app.core.model_loader import model_loader

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Returns API status and model info.
    Use this to verify the service is running before sending predictions.
    """
    return HealthResponse(
        status       = "ok" if model_loader.is_loaded else "model_not_loaded",
        model_loaded = model_loader.is_loaded,
        model_name   = settings.MODEL_NAME,
        device       = settings.DEVICE,
        version      = settings.VERSION,
    )
