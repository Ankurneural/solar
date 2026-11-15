"""Model(s) -> per-head probability maps for one preprocessed frame.

Mirrors the training sampling mode so train and test see the same scale:
  full  resize the frame to `input_size`, one forward pass (+ flip TTA)
  crop  scale by `scale`, sliding window of `crop_size` tiles, Hann blending
Several models (folds / architectures) are averaged in probability space.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from .dataset import normalize
from .resources import GpuThrottle
from .tiling import hann2d, iter_tiles

_throttle = GpuThrottle()

# (flip dims, transpose H<->W). "d4" = all 8 symmetries of the square.
_TTA = {
    "none": [((), False)],
    "hflip": [((), False), ((3,), False)],
    "flip4": [((), False), ((3,), False), ((2,), False), ((2, 3), False)],
}
_TTA["d4"] = _TTA["flip4"] + [(d, True) for d, _ in _TTA["flip4"]]


def amp_dtype() -> torch.dtype:
    """bf16 where the GPU supports it (Ampere+, e.g. RTX 4090); fp16 on older GPUs (Kaggle T4/P100)."""
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


_AMP_OK = [True]  # switched off for the rest of the run if a mixed-precision forward pass fails


def amp_forward(model, x: torch.Tensor, enabled: bool):
    """model(x) under autocast; if cuDNN cannot run the mixed-precision kernels (seen with fp16 on a T4),
    warn once and continue in fp32."""
    if enabled and _AMP_OK[0]:
        try:
            with torch.autocast(x.device.type, dtype=amp_dtype()):
                return model(x)
        except RuntimeError as e:
            _AMP_OK[0] = False
            print(f"[inference] mixed precision failed ({str(e).splitlines()[0]}); continuing in fp32", flush=True)
            torch.cuda.empty_cache()
    return model(x)


@torch.no_grad()
def _forward_tta(models, x: torch.Tensor, tta: str, amp: bool) -> torch.Tensor:
    acc, n = None, 0
    for m in models:
        for dims, tr in _TTA[tta]:
            _throttle.begin()
            xi = x.transpose(2, 3) if tr else x
            xi = torch.flip(xi, dims) if dims else xi
            p = torch.sigmoid(amp_forward(m, xi, amp).float())
            p = torch.flip(p, dims) if dims else p
            p = p.transpose(2, 3) if tr else p
            acc = p if acc is None else acc + p
            n += 1
            _throttle.step()
    return acc / n


@torch.no_grad()
def predict_probs(models, img_u8: np.ndarray, cfg: dict, device: str) -> dict[str, np.ndarray]:
    """-> {head: float32 HxW} at the model's working resolution."""
    d, inf = cfg["data"], cfg.get("infer", {})
    heads = cfg["model"]["heads"]
    tta = inf.get("tta", "hflip")
    amp = device == "cuda"

    if d["mode"] == "full":
        s = d["input_size"]
        im = cv2.resize(img_u8, (s, s), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(normalize(im))[None, None].to(device)
        p = _forward_tta(models, x, tta, amp)[0].cpu().numpy()
    else:
        size = round(img_u8.shape[0] * d.get("scale", 1.0))
        im = img_u8 if size == img_u8.shape[0] else cv2.resize(
            img_u8, (size, size), interpolation=cv2.INTER_AREA)
        tile = d["crop_size"]
        overlap = inf.get("overlap", tile // 4)
        win = hann2d(tile)
        acc = np.zeros((len(heads), size, size), np.float32)
        wsum = np.zeros((size, size), np.float32)
        x_all = normalize(im)
        boxes = list(iter_tiles(size, size, tile, overlap))
        bs = inf.get("batch_size", 4)
        for i in range(0, len(boxes), bs):
            chunk = boxes[i:i + bs]
            x = torch.from_numpy(np.stack([x_all[y0:y1, x0:x1] for y0, x0, y1, x1 in chunk]))
            p = _forward_tta(models, x[:, None].to(device), tta, amp).cpu().numpy()
            for (y0, x0, y1, x1), pi in zip(chunk, p):
                acc[:, y0:y1, x0:x1] += pi * win
                wsum[y0:y1, x0:x1] += win
        p = acc / np.clip(wsum, 1e-6, None)
    return {h: p[i] for i, h in enumerate(heads)}


def load_models(ckpt_paths: list[str], device: str):
    from .model import build_model

    models, cfg = [], None
    for path in ckpt_paths:
        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = cfg or ck["cfg"]
        m = build_model(ck["cfg"], pretrained=False)
        m.load_state_dict(ck.get("ema") or ck["model"])
        models.append(m.to(device).eval())
    return models, cfg
