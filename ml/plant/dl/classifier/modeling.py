"""Model factory helpers."""

import timm
import torch.nn as nn


def create_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
