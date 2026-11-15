"""Rotated refiner crop geometry: crop -> paste round trip and target alignment."""
import cv2
import numpy as np

from src.refiner import (NATIVE, ROT_H, ROT_W, box_inputs, extract_affine, jitter_affine, make_box,
                         paste_affine, refined_native, rot_box)


def _ellipse(size, angle, axes=(300, 25), centre=(900, 1100)):
    m = np.zeros((size, size), np.uint8)
    s = size / NATIVE
    cv2.ellipse(m, (int(centre[0] * s), int(centre[1] * s)), (int(axes[0] * s), int(axes[1] * s)),
                angle, 0, 360, 1, -1)
    return m


def _iou(a, b):
    return (a & b).sum() / max(1, (a | b).sum())


def test_rot_box_aligns_long_axis():
    for angle in (0, 30, 75, 120):
        seed = _ellipse(1024, angle)
        crop = extract_affine(seed.astype(np.float32), rot_box(seed)) > 0.5
        ys, xs = np.nonzero(crop)
        assert np.ptp(xs) > 3 * np.ptp(ys), angle  # long axis horizontal
        assert abs(xs.mean() - ROT_W / 2) < 8 and abs(ys.mean() - ROT_H / 2) < 8


def test_round_trip_native():
    for angle in (10, 45, 100):
        native = _ellipse(NATIVE, angle).astype(bool)
        A = rot_box(_ellipse(1024, angle))
        crop = extract_affine(native.astype(np.float32), A)
        back = refined_native(crop, A)
        assert _iou(back, native) > 0.9, angle


def test_jitter_keeps_affine_shape_and_inputs():
    seed = _ellipse(1024, 30)
    A = jitter_affine(rot_box(seed), np.random.RandomState(0))
    assert A.shape == (2, 3)
    img = np.random.RandomState(1).randint(0, 255, (NATIVE, NATIVE), np.uint8)
    x = box_inputs(img, seed.astype(np.float32), seed, A)
    assert x.shape == (3, ROT_H, ROT_W)


def test_square_box_unchanged():
    seed = _ellipse(1024, 30)
    box = make_box("square", (500.0, 900.0, 1300.0, 1300.0), seed)
    assert isinstance(box, tuple) and len(box) == 3


def test_paste_off_frame_is_clipped():
    seed = _ellipse(1024, 20, centre=(30, 1000))
    A = rot_box(seed)
    y0, x0, p = paste_affine(np.ones((ROT_H, ROT_W), np.float32), A)
    assert x0 == 0 and y0 >= 0 and p.shape[1] <= NATIVE
