"""Generate a tiny fake MAGFiLO-format dataset for smoke-testing the pipeline
before (or without) the real data. Same folder layout, JSON schema, image size
and naming as the competition data; contents are crude: a limb-darkened disk
with dark, curvy, elongated blobs annotated by 1-3 jittery "annotators".

  python make_synthetic_data.py --out data/synthetic --n-train 24 --n-test 4
"""
from __future__ import annotations

import argparse
import json
import os
import uuid

import cv2
import numpy as np

SIZE = 2048
SITES = ["Bh", "Ch", "Lh", "Mh", "Th", "Uh"]


def make_frame(rng):
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    cx, cy, r = SIZE / 2 + rng.normal(0, 8), SIZE / 2 + rng.normal(0, 8), rng.uniform(880, 940)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
    img = np.where(rr < 1, 170 * (0.55 + 0.45 * np.sqrt(np.clip(1 - rr ** 2, 0, 1))), 8)
    img = img * (1 + 0.05 * cv2.GaussianBlur(rng.normal(0, 1, (SIZE, SIZE)).astype(np.float32), (0, 0), 25))
    fil = []
    for _ in range(rng.integers(2, 7)):
        # Random smooth spine inside the disk.
        a, rad = rng.uniform(0, 2 * np.pi), rng.uniform(0, 0.75) * r
        p0 = np.array([cx + rad * np.cos(a), cy + rad * np.sin(a)])
        ang, n = rng.uniform(0, 2 * np.pi), rng.integers(6, 14)
        pts = [p0]
        for _ in range(n):
            ang += rng.normal(0, 0.35)
            pts.append(pts[-1] + 18 * np.array([np.cos(ang), np.sin(ang)]))
        spine = np.array(pts)
        width = rng.uniform(5, 14)
        m = np.zeros((SIZE, SIZE), np.uint8)
        cv2.polylines(m, [np.round(spine).astype(np.int32)], False, 1, int(width))
        img[m > 0] *= rng.uniform(0.6, 0.8)
        fil.append((m, spine))
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.5) + rng.normal(0, 3, (SIZE, SIZE))
    return np.clip(img, 0, 255).astype(np.uint8), fil


def polygon_of(mask, rng, jitter):
    m = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)[:, 0].astype(np.float64)
    c += rng.normal(0, jitter, c.shape)
    c = np.vstack([c, c[:1]])  # closed path: first point repeated, like MAGFiLO
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n-train", type=int, default=24)
    ap.add_argument("--n-test", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    root = os.path.join(args.out, "MAGFiLO_1.0_Kaggle_2026")
    tr_dir, te_dir = os.path.join(root, "train", "train_images"), os.path.join(root, "test", "test_images")
    os.makedirs(tr_dir, exist_ok=True)
    os.makedirs(te_dir, exist_ok=True)

    images, anns = [], []
    for i in range(args.n_train + args.n_test):
        day = 1 + i % 28
        name = f"2016{1 + i // 28:02d}{day:02d}{rng.integers(0, 24):02d}0000{SITES[i % 6]}"
        img, fils = make_frame(rng)
        test = i >= args.n_train
        cv2.imwrite(os.path.join(te_dir if test else tr_dir, f"{name}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        if test:
            continue
        for a in range(rng.choice([1, 1, 2, 3])):
            set_id = f"0101{a + 1:02d}-{name}"
            images.append({"id": set_id, "width": SIZE, "height": SIZE, "file_name": f"{name}.jpg",
                           "license": 1, "date_captured": ""})
            for m, spine in fils:
                if rng.random() < 0.1:  # annotators sometimes skip a filament
                    continue
                poly = polygon_of(m, rng, jitter=1.5)
                x0, y0 = poly.min(0)
                x1, y1 = poly.max(0)
                anns.append({"id": str(uuid.UUID(int=int(rng.integers(1 << 62)))), "image_id": set_id,
                             "category_id": int(rng.integers(1, 5)),
                             "segmentation": [poly.ravel().round(2).tolist()],
                             "area": float(m.sum()), "spine": spine.ravel().round(2).tolist(),
                             "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
                             "iscrowd": 0})
    coco = {"info": {"description": "synthetic smoke-test data"}, "images": images,
            "annotations": anns, "licenses": [],
            "categories": [{"id": k, "name": n, "supercategory": "filament"} for k, n in
                           enumerate(["Left", "Right", "Unidentifiable", "Ambiguous"], 1)]}
    with open(os.path.join(root, "train", "MAGFiLO_1.0_Annotations_kaggle2026_train.json"), "w") as f:
        json.dump(coco, f)
    print(f"{args.n_train} train / {args.n_test} test frames, {len(images)} annotation sets, "
          f"{len(anns)} filaments -> {root}")


if __name__ == "__main__":
    main()
