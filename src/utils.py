"""Small shared helpers: config loading (with CLI overrides), seeding, device."""
from __future__ import annotations

import copy
import os
import random
from typing import Any

import numpy as np
import yaml


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML config. A config may name a parent with `inherit: other.yaml`
    (resolved relative to itself) and only list the keys it changes.

    `overrides` are dotted CLI assignments, e.g. ["train.lr=1e-4", "model.encoder=resnet50"];
    values are parsed as YAML so numbers/bools/lists work.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        base = load_config(os.path.join(os.path.dirname(path), parent))
        cfg = _deep_update(base, cfg)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        *parents, leaf = key.strip().split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(raw)
    return cfg


def save_config(cfg: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(copy.deepcopy(cfg), f, sort_keys=False)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def setup_device(verbose: bool = True) -> str:
    """Pick CUDA when visible (enabling cudnn.benchmark + TF32), else CPU.

    On GPUs without bf16 (pre-Ampere, e.g. Kaggle's T4), inference runs in fp16, and cuDNN's benchmark search
    failed there for ConvNeXt's fp16 depthwise convolutions ("FIND was unable to find an engine"), so benchmark
    mode is off and cuDNN's heuristics pick the kernels."""
    import torch

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = torch.cuda.is_bf16_supported()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if verbose:
            i = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(i)
            print(f"[device] CUDA GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)")
        return "cuda"
    if verbose:
        print("[device] No CUDA GPU visible -> CPU (training will be slow).")
    return "cpu"
