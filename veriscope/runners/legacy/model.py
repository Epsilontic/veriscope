# veriscope/runners/legacy/model.py
"""
Model and schedule helpers for legacy CIFAR runners.

Pulled from runners/legacy_cli_refactor.py with identical behavior:
- make_model: ResNet-18 adapted for CIFAR-10 (3x32x32, no maxpool).
- make_opt: SGD optimizer with Nesterov, driven by a cfg dict.
- lr_at: warmup + cosine schedule.
- penult: hook to the penultimate feature layer.
"""

from __future__ import annotations

from typing import Any, Dict

import math
import torch
from torch import nn
import torchvision  # type: ignore[import-untyped]

# CIFAR-10 classes (kept local; caller can override via num_classes if needed)
C: int = 10


def make_model(num_classes: int = C) -> nn.Module:
    """Construct a CIFAR-10–style ResNet-18 (3x32x32, no initial maxpool)."""
    try:
        m = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    except TypeError:
        # Older torchvision uses `pretrained` instead of `weights`
        m = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
    # CIFAR-10 stem tweak: 3x3 conv, stride 1, no maxpool
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def make_opt(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """SGD + Nesterov using hyperparameters from cfg."""
    return torch.optim.SGD(
        model.parameters(),
        lr=float(cfg["base_lr"]),
        momentum=float(cfg["momentum"]),
        weight_decay=float(cfg["weight_decay"]),
        nesterov=True,
    )


def lr_at(epoch: int, total: int, base: float, warmup: int, cosine: bool) -> float:
    """Learning-rate schedule: linear warmup then optional cosine decay."""
    if warmup > 0 and epoch < warmup:
        return base * (epoch + 1) / float(warmup)
    if cosine:
        t = (epoch - warmup) / max(1.0, float(total - warmup))
        return base * 0.5 * (1.0 + math.cos(math.pi * t))
    return base


@torch.no_grad()
def penult(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Penultimate feature extractor for the ResNet-18 backbone."""
    h = model.relu(model.bn1(model.conv1(x)))
    h = model.layer1(h)
    h = model.layer2(h)
    h = model.layer3(h)
    h = model.layer4(h)
    h = model.avgpool(h)
    return torch.flatten(h, 1)