"""Model factory built on segmentation_models_pytorch.

One network, one output channel per head (mask / boundary / spine). Input is a
single grayscale channel: smp adapts the pretrained first conv by summing its
RGB filters, so ImageNet weights still help without faking 3 channels.
"""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn


def build_model(cfg: dict, pretrained: bool = True) -> nn.Module:
    m = cfg["model"]
    return smp.create_model(
        arch=m["arch"],
        encoder_name=m["encoder"],
        encoder_weights=(m.get("encoder_weights") or None) if pretrained else None,
        in_channels=1,
        classes=len(m["heads"]),
        # e.g. {"pretrained_cfg_overlay": {"file": "weights/convnext_tiny_22k_1k_224.pth"}} to load
        # timm (tu-) encoder weights from a local file (Hugging Face is blocked on this network)
        **(m.get("encoder_kwargs") or {}),
    )
