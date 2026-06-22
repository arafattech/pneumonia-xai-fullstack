# =============================================================
# app/ml/model.py  —  DenseNet121 / ResNet50 / EfficientNet-B0
# =============================================================

import torch
import torch.nn as nn
from torchvision import models


def _head(in_features: int, num_classes: int) -> nn.Sequential:
    """Shared classifier head for all backbones."""
    return nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )


def build_model(
    name: str        = "densenet121",
    num_classes: int = 2,
    pretrained: bool = True,
) -> tuple[nn.Module, nn.Module]:
    """
    Build transfer-learning model.

    Returns
    -------
    (model, target_layer_for_gradcam)
    """
    w = "IMAGENET1K_V1" if pretrained else None

    if name == "densenet121":
        m = models.densenet121(weights=w)
        m.classifier = _head(m.classifier.in_features, num_classes)
        tl = m.features.denseblock4.denselayer16.conv2

    elif name == "resnet50":
        m = models.resnet50(weights=w)
        m.fc = _head(m.fc.in_features, num_classes)
        tl = m.layer4[-1].conv3

    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=w)
        m.classifier = _head(m.classifier[1].in_features, num_classes)
        tl = m.features[-1][0]

    else:
        raise ValueError(f"Unknown model: {name}")

    return m, tl
