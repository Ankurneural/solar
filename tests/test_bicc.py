"""BiCC loss: one equal-weight term per consensus component and per false-positive blob."""
import time

import torch

from src.losses import bicc_loss


def _tile():
    t = torch.zeros(1, 64, 64)
    t[0, 5:15, 5:40] = 1.0  # large filament
    t[0, 40:43, 40:46] = 1.0  # small filament
    return t


def test_perfect_prediction_is_near_zero():
    t = _tile()
    assert bicc_loss(t.clone(), t).item() < 0.01


def test_missing_small_object_costs_as_much_as_large():
    t = _tile()
    miss_small, miss_large = t.clone(), t.clone()
    miss_small[0, 40:43, 40:46] = 0
    miss_large[0, 5:15, 5:40] = 0
    a, b = bicc_loss(miss_small, t).item(), bicc_loss(miss_large, t).item()
    assert abs(a - b) < 0.05 and a > 0.4  # 2 * 0.5 * 1 / 2 objects = 0.5


def test_false_positive_blob_and_gradient():
    t = _tile()
    p = t.clone()
    p[0, 55:60, 5:10] = 0.9  # blob touching no consensus pixel
    p.requires_grad_(True)
    loss = bicc_loss(p, t)
    # 2 * (0.5 * (GT terms ~0.03 + ~0.38: the blob lies in the small filament's Voronoi cell) + 0.5 * 0.9) / 3 objects
    assert 0.35 < loss.item() < 0.5
    loss.backward()
    assert p.grad[0, 57, 7] > 0  # pushes the FP blob down


def test_empty_tile():
    z = torch.zeros(2, 32, 32, requires_grad=True)
    assert bicc_loss(torch.sigmoid(z), torch.zeros(2, 32, 32)).item() == 0.0


def test_speed_training_batch():
    torch.manual_seed(0)
    t = (torch.rand(8, 512, 512) > 0.995).float()
    t = torch.nn.functional.max_pool2d(t[:, None], 9, 1, 4)[:, 0]
    p = torch.rand(8, 512, 512)
    t0 = time.perf_counter()
    bicc_loss(p, t)
    assert time.perf_counter() - t0 < 2.0
