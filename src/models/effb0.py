# src/models/effb0.py
from __future__ import annotations

import torch
import torch.nn as nn
import timm

class SimpleEffB0(nn.Module):
    def __init__(
        self,
        pretrained: bool = False,
        in_chans: int = 3,
        num_classes: int = 1,      # must be 1 to match your checkpoint
        drop_rate: float = 0.2,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=num_classes,   # <- keep this 1
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x):
        return self.backbone(x)

class EffB0Detector(nn.Module):
    """
    Binary deepfake image detector with an EfficientNet-B0 backbone.

    - Outputs raw logits of shape (B, num_classes). For binary detection use
      num_classes=1 and BCEWithLogitsLoss.
    - Set `pretrained=True` to use ImageNet weights.
    - `drop_rate` / `drop_path_rate` apply to the backbone (timm).
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 1,
        drop_rate: float = 0.2,
        drop_path_rate: float = 0.1,
        in_chans: int = 3,
    ):
        super().__init__()

        # Feature extractor without classifier head (num_classes=0 returns features)
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_chans,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # Feature dimension (timm models usually expose .num_features)
        in_feats = getattr(self.backbone, "num_features", None)
        if in_feats is None:
            # Fallback for unusual timm versions
            try:
                in_feats = self.backbone.get_classifier().in_features  # type: ignore[attr-defined]
            except Exception as e:
                raise AttributeError(
                    "Could not infer feature dimension from EfficientNet-B0 backbone."
                ) from e

        # Small classification head
        self.head = nn.Sequential(
            nn.Linear(in_feats, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)          # (B, in_feats)
        logits = self.head(feats)         # (B, num_classes)
        return logits
