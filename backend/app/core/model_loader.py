# =============================================================
# app/core/model_loader.py  —  Singleton: load model once
# =============================================================

import os
import torch
import torch.nn as nn

from app.core.config import settings
from app.ml.model import build_model


class ModelLoader:
    """
    Loads and caches the model + target layer for Grad-CAM.
    Call .load() once at startup; then access .model and .target_layer.
    """

    def __init__(self):
        self._model:        nn.Module | None = None
        self._target_layer: nn.Module | None = None
        self._device = torch.device(settings.DEVICE)

    def load(self) -> None:
        """Build model and load saved weights if they exist."""
        self._model, self._target_layer = build_model(
            name        = settings.MODEL_NAME,
            num_classes = settings.NUM_CLASSES,
            pretrained  = False,          # weights come from .pth file
        )

        weights_path = os.path.abspath(settings.MODEL_PATH)

        if os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=self._device)
            # support both raw state_dict and checkpoint dicts
            if isinstance(state, dict) and "model_state" in state:
                state = state["model_state"]
            self._model.load_state_dict(state)
            print(f"    Weights loaded from: {weights_path}")
        else:
            print(f"    ⚠️  No weights at {weights_path} — using random init.")
            print("    ⚠️  Train the model first and place best_model.pth there.")

        self._model = self._model.to(self._device)
        self._model.eval()

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call model_loader.load() first.")
        return self._model

    @property
    def target_layer(self) -> nn.Module:
        if self._target_layer is None:
            raise RuntimeError("Target layer not set. Call model_loader.load() first.")
        return self._target_layer

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def is_loaded(self) -> bool:
        return self._model is not None


# Global singleton
model_loader = ModelLoader()
