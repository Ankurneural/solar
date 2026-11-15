"""Pick the best tuned blend and, if it beats the reference, build its stack and upload it (unattended).

  python tools/auto_stack.py --cands runs/w13_a runs/w13_b --ref-pp 0.4527 --ref-stack 0.4667 \
      --refiners runs/w71_b1_r34/refiner_rot --wait-job b1b_s2 --msg "v12: ..."

Steps: best `postprocess_tuned.yaml` OOF PQ among --cands; stop unless it beats --ref-pp by --margin. Then (CPU) train_scorer,
the refiner candidate table, copy the borrowed refiners; wait until the GPU job --wait-job is DONE (never two GPU jobs);
(GPU) tune_stack + D4 inference; upload with tools/submit.py if the stack OOF beats --ref-stack.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def run(*args):
    print(">", " ".join(args), flush=True)
    subprocess.run([PY, *args], cwd=ROOT, check=True)


def wait_job(name: str):
    path = os.path.join(ROOT, "runs", "jobs", f"{name}.status")
    while True:
        s = open(path).read().strip() if os.path.exists(path) else ""
        if s.startswith("DONE"):
            print(f"waited for {name}: {s}", flush=True)
            return
        time.sleep(60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", nargs="+", required=True)
    ap.add_argument("--ref-pp", type=float, required=True, help="tuned threshold OOF PQ of the current best blend")
    ap.add_argument("--ref-stack", type=float, required=True, help="stack OOF PQ of the current best submission")
    ap.add_argument("--margin", type=float, default=0.0005)
    ap.add_argument("--refiners", required=True, help="dir with fold*.pt refiners to borrow")
    ap.add_argument("--wait-job", default=None, help="GPU job to wait for before tune_stack/infer")
    ap.add_argument("--msg", required=True)
    args = ap.parse_args()

    scores = {}
    for c in args.cands:
        p = os.path.join(ROOT, c, "postprocess_tuned.yaml")
        if os.path.exists(p):
            scores[c] = yaml.safe_load(open(p))["oof_metrics"]["pq"]
    for c, s in sorted(scores.items(), key=lambda kv: -kv[1]):
        print(f"{c}: {s:.4f}")
    if not scores:
        if args.wait_job:
            wait_job(args.wait_job)
        sys.exit("no tuned candidates")
    best = max(scores, key=scores.get)
    if scores[best] <= args.ref_pp + args.margin:
        print(f"best {best} {scores[best]:.4f} does not beat {args.ref_pp:.4f} + {args.margin}: no stack built")
        # still wait: jobs queued behind this one assume the GPU is free when it finishes
        if args.wait_job:
            wait_job(args.wait_job)
        return
    run("train_scorer.py", "--run", best)
    run("train_refiner.py", "--run", best, "--table-only")
    for f in glob.glob(os.path.join(ROOT, args.refiners, "fold*.pt")):
        shutil.copy(f, os.path.join(ROOT, best, "refiner"))
    if args.wait_job:
        wait_job(args.wait_job)
    run("tune_stack.py", "--run", best)
    q = yaml.safe_load(open(os.path.join(ROOT, best, "stack.yaml")))["oof_pq"]
    print(f"{best} stack OOF {q:.4f} vs {args.ref_stack:.4f}", flush=True)
    if q <= args.ref_stack:
        print("no stack gain: not uploading")
        return
    csv = f"{best}/submission_{os.path.basename(best)}_stack.csv"
    run("infer.py", "--run", best, "--tta", "d4", "--out", csv)
    run("tools/submit.py", csv, "--upload", "-m", f"{args.msg} ({os.path.basename(best)}), OOF PQ {q:.4f}")


if __name__ == "__main__":
    main()
