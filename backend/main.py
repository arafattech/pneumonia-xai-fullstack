# =============================================================
# main.py  —  FastAPI application entry point
# =============================================================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.api import predict, xai, health
from app.core.config import settings
from app.core.model_loader import model_loader
from app.db.database import init_db

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="Pneumonia XAI API",
    description=(
        "Explainable AI-based Pneumonia Detection from Chest X-ray Images. "
        "Uses DenseNet121 with Grad-CAM and LIME explanations."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────
app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, tags=["Prediction"])
app.include_router(xai.router, prefix="/explain", tags=["Explainability"])


# ── Startup: load model once ──────────────────────────────────
@app.on_event("startup")
async def startup_event():
    print("🚀  Starting Pneumonia XAI API...")
    init_db()
    print("✅  Database tables ready")
    model_loader.load()
    print(f"✅  Model loaded on {settings.DEVICE}")
    print(f"✅  API ready at http://0.0.0.0:{settings.PORT}")


@app.on_event("shutdown")
async def shutdown_event():
    print("🛑  Shutting down API...")


# ── Root ──────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Pneumonia XAI API is running",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "explain_gradcam": "/explain/gradcam",
        "explain_lime": "/explain/lime",
    }
