"""Albumentations pipelines for single-channel full-disk H-alpha frames.

The Sun has no preferred orientation in the image plane (solar north varies with
date/site), so flips and arbitrary rotations are label-preserving. Photometric
augmentation mimics what differs between GONG sites and days: contrast, gamma,
haze/clouds (low-frequency brightness), seeing blur and JPEG noise. We avoid
anything that changes the disk's shape (elastic, heavy shear): annotators'
polygons follow real structures.
"""
from __future__ import annotations

import albumentations as A
import cv2


def train_transforms(cfg: dict) -> A.Compose:
    a = cfg.get("augment", {})
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(rotate=(-180, 180), scale=tuple(a.get("scale", [0.9, 1.1])),
                 translate_percent=(-0.03, 0.03), interpolation=cv2.INTER_LINEAR,
                 mask_interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT,
                 fill=0, fill_mask=0, p=a.get("affine_p", 0.7)),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.25, p=0.7),
        A.RandomGamma(gamma_limit=(80, 125), p=0.4),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.Defocus(radius=(1, 3)),
        ], p=a.get("blur_p", 0.3)),
        A.OneOf([
            A.GaussNoise(std_range=(0.01, 0.04)),
            A.ImageCompression(quality_range=(40, 90)),
        ], p=a.get("noise_p", 0.3)),
    ])


def val_transforms(cfg: dict) -> A.Compose | None:
    return None
