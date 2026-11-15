"""Grid-search post-processing parameters on out-of-fold probability maps.

  python tune_postprocess.py --run runs/baseline
  python tune_postprocess.py --run runs/baseline --eval-size 2048   # exact, slower

Every OOF image was predicted by a model that never saw it, so the tuned
parameters transfer to the test set. Search is coordinate-wise (one parameter
at a time, a few sweeps) rather than a full grid, which keeps it to minutes.
Writes <run>/postprocess_tuned.yaml and prints the PQ breakdown.
"""
from __future__ import annotations

import argparse
import copy
import glob
import os

import yaml

from src.coco_io import load_observations
from src.evaluation import load_disks, load_probs, score_obs
from src.utils import load_config
from src.resources import limit_resources

# Pixel parameters are in native 2048-px units (see src/postprocess.py).
SEARCH = {
    "threshold": [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65],
    "close_px": [0, 4, 8, 12, 20],
    "merge_px": [0, 10, 20, 40, 60],
    "min_area": [0, 100, 200, 400, 600, 900, 1400],
    "min_score": [0.0, 0.6, 0.7, 0.8, 0.9],
    "grow_px": [0, 2, 4, 6, 10],
}


def main():
    limit_resources(gpu=False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run dir (default: paths.out_dir of --config)")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--eval-size", type=int, default=1024)
    ap.add_argument("--sweeps", type=int, default=2)
    args = ap.parse_args()
    args.run = args.run or load_config(args.config)["paths"]["out_dir"]

    cfg_path = sorted(glob.glob(os.path.join(args.run, "fold*", "config_used.yaml")))[0]
    cfg = load_config(cfg_path)
    d, heads = cfg["data"], cfg["model"]["heads"]
    obs = load_observations(d["train_json"])
    disks = load_disks(d["cache_dir"])

    probs = {}
    for oof_dir in sorted(glob.glob(os.path.join(args.run, "fold*", "oof"))):
        for p in glob.glob(os.path.join(oof_dir, f"*_{heads[0]}.png")):
            oid = os.path.basename(p).rsplit("_", 1)[0]
            probs[oid] = load_probs(oof_dir, oid, heads, as_uint8=True)
    print(f"{len(probs)} OOF observations from {args.run}")

    space = dict(SEARCH)
    if "spine" in heads:
        space["grouping"] = ["cc", "spine"]
        space["spine_threshold"] = [0.3, 0.4, 0.5, 0.6]
    if "boundary" in heads:
        space["boundary_cut"] = [0, 0.5, 0.6, 0.7, 0.8]

    pp = copy.deepcopy(cfg["postprocess"])
    best = score_obs(obs, probs, disks, pp, args.eval_size)
    print(f"start PQ {best['pq']:.4f} with {pp}")
    for sweep in range(args.sweeps):
        for key, values in space.items():
            for v in values:
                if pp.get(key) == v:
                    continue
                trial = {**pp, key: v}
                m = score_obs(obs, probs, disks, trial, args.eval_size)
                if m["pq"] > best["pq"] + 1e-5:
                    best, pp = m, trial
                    print(f"  sweep {sweep} {key}={v}: PQ {m['pq']:.4f} "
                          f"(SQ {m['sq']:.3f} RQ {m['rq']:.3f} FP {m['fp']} FN {m['fn']})")

    print(f"\nbest PQ {best['pq']:.4f}  SQ {best['sq']:.3f}  RQ {best['rq']:.3f}  "
          f"TP {best['tp']} FP {best['fp']} FN {best['fn']}")
    out = os.path.join(args.run, "postprocess_tuned.yaml")
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump({"postprocess": pp, "oof_metrics": best, "eval_size": args.eval_size}, f,
                       sort_keys=False)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
