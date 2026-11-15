"""Stage-1 targets: with data.target_offset -0.5 the cv2 rasterization matches the organizers' pycocotools masks."""
import numpy as np

from src.coco_io import Filament, Observation, polygon_to_rle, rle_to_mask
from src.targets import rasterize


def _obs(size=2048, seed=0):
    rng = np.random.default_rng(seed)
    fils = []
    for _ in range(12):  # thin rotated quads with sub-pixel vertices, like filaments
        c = rng.uniform(300, size - 300, 2)
        th = rng.uniform(0, np.pi)
        u, v = np.array([np.cos(th), np.sin(th)]), np.array([-np.sin(th), np.cos(th)])
        L, W = rng.uniform(40, 150), rng.uniform(4, 12)
        poly = np.array([c - L * u - W * v, c + L * u - W * v, c + L * u + W * v, c - L * u + W * v])
        fils.append(Filament(poly, None, 0, 0.0))
    o = Observation("x", "x.png", size, size, {"a": fils})
    return o


def _iou(a, b):
    return (a & b).sum() / (a | b).sum()


def test_offset_matches_pycocotools():
    o = _obs()
    for size in (2048, 1024):
        s = size / 2048
        ref = np.zeros((size, size), bool)
        for f in o.annotations["a"]:
            ref |= rle_to_mask(polygon_to_rle(f.polygon, size, size, s)).astype(bool)
        old = rasterize(o, size, ["mask"])[0] > 0.5
        new = rasterize(o, size, ["mask"], offset=-0.5)[0] > 0.5
        assert _iou(new, ref) > _iou(old, ref) + 0.02, size
        assert _iou(new, ref) > 0.85, size
