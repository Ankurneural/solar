"""Validate a submission CSV and (optionally) upload it to Kaggle.

  python tools/submit.py runs/X/submission.csv                       # validate only
  python tools/submit.py runs/X/submission.csv --upload -m "message"  # validate, upload, wait for the score

Checks: columns, unique filament ids, ids of the form <obs_id>_<k>, no quotes,
decodable 2048x2048 COCO RLE counts, no empty masks, masks disjoint within each
frame. Uploads go through tools/kaggle_cli.py (token redacted). The public score
is appended to runs/jobs/submissions.csv.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMP = "filament-segmentation-2026"
KCLI = [sys.executable, os.path.join(ROOT, "tools", "kaggle_cli.py")]


def validate(path: str) -> list[str]:
    errors = []
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(df.columns) != ["filament_id", "segmentation_rle"]:
        return [f"columns {list(df.columns)}"]
    if df.filament_id.duplicated().any():
        errors.append(f"{df.filament_id.duplicated().sum()} duplicate ids")
    bad = ~df.filament_id.str.match(r"^\d{14}[A-Za-z]{2}_\d+$")
    if bad.any():
        errors.append(f"{bad.sum()} malformed ids, e.g. {df.filament_id[bad].iloc[0]}")
    if df.segmentation_rle.str.contains("[\"']").any():
        errors.append("quotes in RLE")
    obs = df.filament_id.str.split("_", n=1).str[0]
    n_empty = n_overlap = 0
    for _, g in df.groupby(obs):
        rles = [{"size": [2048, 2048], "counts": c.encode()} for c in g.segmentation_rle]
        areas = mask_util.area(rles)
        n_empty += int((areas == 0).sum())
        acc = np.zeros((2048, 2048), np.uint8)
        for m in mask_util.decode(rles).transpose(2, 0, 1):
            acc += m
        n_overlap += int((acc > 1).any())
    if n_empty:
        errors.append(f"{n_empty} empty masks")
    if n_overlap:
        errors.append(f"{n_overlap} frames with overlapping masks")
    print(f"{path}: {len(df)} filaments on {obs.nunique()} frames; "
          f"{'OK' if not errors else 'ERRORS: ' + '; '.join(errors)}")
    return errors


def upload(path: str, message: str) -> None:
    r = subprocess.run(KCLI + ["competitions", "submit", "-c", COMP, "-f", path, "-m", message],
                       capture_output=True, text=True)
    print(r.stdout[-300:], r.stderr[-300:])
    if r.returncode or "Successfully" not in r.stdout:
        raise SystemExit("upload failed")
    name = os.path.basename(path)
    for _ in range(60):  # scoring takes a few minutes
        time.sleep(30)
        out = subprocess.run(KCLI + ["competitions", "submissions", "-c", COMP, "--csv"],
                             capture_output=True, text=True).stdout
        rows = [l for l in out.splitlines() if l.split(",")[1:2] == [name]]
        if rows and "COMPLETE" in rows[0]:
            score = re.findall(r"COMPLETE,([0-9.]*)", rows[0])
            score = score[0] if score else "?"
            print(f"public score: {score}")
            log = os.path.join(ROOT, "runs", "jobs", "submissions.csv")
            os.makedirs(os.path.dirname(log), exist_ok=True)
            new = not os.path.exists(log)
            with open(log, "a", encoding="utf-8") as f:
                if new:
                    f.write("time,file,message,public_score\n")
                f.write(f"{time.strftime('%Y-%m-%d %H:%M')},{path},\"{message}\",{score}\n")
            return
        if rows and "ERROR" in rows[0]:
            raise SystemExit(f"Kaggle rejected the submission: {rows[0]}")
    print("still scoring; check `python tools/kaggle_cli.py competitions submissions -c " + COMP + "`")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("-m", "--message", default="")
    args = ap.parse_args()
    if validate(args.csv):
        raise SystemExit(1)
    if args.upload:
        upload(args.csv, args.message or os.path.basename(args.csv))


if __name__ == "__main__":
    main()
