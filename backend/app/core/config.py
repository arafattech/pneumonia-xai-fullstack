# =============================================================
# app/core/config.py  —  Application settings via pydantic
# =============================================================

import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── API ───────────────────────────────────────────────────
    APP_NAME: str        = "Pneumonia XAI API"
    VERSION:  str        = "1.0.0"
    PORT:     int        = 8000
    DEBUG:    bool       = False

    # ── CORS ──────────────────────────────────────────────────
    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ]

    # ── Model ─────────────────────────────────────────────────
    MODEL_PATH: str  = os.path.join(
        os.path.dirname(__file__), "..", "ml", "weights", "best_model.pth"
    )
    MODEL_NAME: str  = "densenet121"   # densenet121 | resnet50 | efficientnet_b0
    NUM_CLASSES: int = 2
    CLASS_NAMES: list[str] = ["NORMAL", "PNEUMONIA"]
    DEVICE: str      = "cpu"           # "cuda" if GPU available

    # ── Image ─────────────────────────────────────────────────
    IMG_SIZE:       int   = 224
    MAX_FILE_SIZE:  int   = 10 * 1024 * 1024   # 10 MB
    ALLOWED_TYPES: list[str] = ["image/jpeg", "image/png", "image/jpg"]

    # ── ImageNet normalisation ────────────────────────────────
    IMAGENET_MEAN: list[float] = [0.485, 0.456, 0.406]
    IMAGENET_STD:  list[float] = [0.229, 0.224, 0.225]

    # ── Database ──────────────────────────────────────────────
    DB_HOST:     str = "localhost"
    DB_PORT:     int = 3306
    DB_USER:     str = "root"
    DB_PASSWORD: str = "root"
    DB_NAME:     str = "pneumonia_xai"

    @property
    def DATABASE_URL(self) -> str:
        return f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
