# =============================================================
# ml_pipeline/src/model.py  —  Transfer learning model
# =============================================================

import torch
import torch.nn as nn
from torchvision import models
import config


def _head(in_feat, num_classes):
    return nn.Sequential(
        nn.BatchNorm1d(in_feat),
        nn.Dropout(0.4),
        nn.Linear(in_feat, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )


def build_model(name=config.MODEL_NAME, num_classes=config.NUM_CLASSES,
                pretrained=config.PRETRAINED):
    w = 'IMAGENET1K_V1' if pretrained else None
    print(f'🔨  Building {name} (pretrained={pretrained})')

    if name == 'densenet121':
        m  = models.densenet121(weights=w)
        m.classifier = _head(m.classifier.in_features, num_classes)
        tl = m.features.denseblock4.denselayer16.conv2

    elif name == 'resnet50':
        m  = models.resnet50(weights=w)
        m.fc = _head(m.fc.in_features, num_classes)
        tl = m.layer4[-1].conv3

    elif name == 'efficientnet_b0':
        m  = models.efficientnet_b0(weights=w)
        m.classifier = _head(m.classifier[1].in_features, num_classes)
        tl = m.features[-1][0]

    else:
        raise ValueError(f'Unknown: {name}')

    tp = sum(p.numel() for p in m.parameters())
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f'    Params total/trainable: {tp:,} / {tr:,}')
    return m, tl


def load_model(path, name=config.MODEL_NAME):
    model, tl = build_model(name, pretrained=False)
    state = torch.load(path, map_location=config.DEVICE)
    if 'model_state' in state:
        state = state['model_state']
    model.load_state_dict(state)
    model = model.to(config.DEVICE)
    model.eval()
    print(f'✅  Weights loaded: {path}')
    return model, tl
