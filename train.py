"""Train one cross-validation fold.

  python train.py --config configs/baseline.yaml --fold 0
  python train.py --config configs/baseline.yaml --fold all       # every fold in turn

Model selection uses the competition metric (PQ after post-processing) on the
held-out fold, not the loss: a lower loss can still mean more fragmented
instances. After training, the best checkpoint predicts its validation fold and
saves the probability maps (out-of-fold, "OOF") that tune_postprocess.py uses.

Outputs in <out_dir>/fold<k>/: best.pt, last.pt, metrics.csv, oof/<obs>_<head>.png
Re-running resumes from last.pt automatically (pass --fresh to start over).
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time

if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

import pandas as pd
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.coco_io import load_observations
from src.dataset import FilamentDataset, load_cached
from src.evaluation import load_disks, save_probs, score_obs
from src.inference import predict_probs
from src.losses import MultiHeadLoss
from src.model import build_model
from src.transforms import train_transforms
from src.resources import GpuThrottle, limit_resources, worker_budget, worker_init
from src.utils import load_config, save_config, seed_everything, setup_device


def make_scheduler(optimizer, tr, steps_per_epoch):
    total = tr["epochs"] * steps_per_epoch
    warm = tr.get("warmup_epochs", 1) * steps_per_epoch

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        prog = (step - warm) / max(1, total - warm)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def predict_fold(model, val_obs, cfg, device) -> dict:
    model.eval()
    cache = cfg["data"]["cache_dir"]
    return {o.obs_id: predict_probs([model], load_cached(cache, o.obs_id), cfg, device)
            for o in tqdm(val_obs, desc="val-predict", leave=False)}


def train_fold(cfg: dict, fold: int, fresh: bool) -> dict:
    seed_everything(cfg["seed"] + fold)
    device = setup_device(verbose=(fold == 0 or fold is None))
    out_dir = os.path.join(cfg["paths"]["out_dir"], f"fold{fold}")
    os.makedirs(out_dir, exist_ok=True)
    save_config(cfg, os.path.join(out_dir, "config_used.yaml"))

    d, tr = cfg["data"], cfg["train"]
    obs = load_observations(d["train_json"])
    folds = pd.read_csv(os.path.join(d["cache_dir"], "folds.csv"))
    disks = load_disks(d["cache_dir"])
    tr_ids = folds.obs_id[folds.fold != fold].tolist()
    va_ids = folds.obs_id[folds.fold == fold].tolist()
    if d.get("max_train"):  # quick experiments
        tr_ids = tr_ids[: d["max_train"]]
    train_obs = [obs[i] for i in tr_ids]
    val_obs = [obs[i] for i in va_ids]
    print(f"[fold {fold}] train {len(train_obs)} obs | val {len(val_obs)} obs")

    train_ds = FilamentDataset(train_obs, cfg, train_transforms(cfg), train=True)
    nw = worker_budget(tr.get("num_workers", 4))
    loader_kw = dict(num_workers=nw, pin_memory=(device == "cuda"), worker_init_fn=worker_init)
    if nw > 0:
        loader_kw.update(persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(train_ds, batch_size=tr["batch_size"], shuffle=True,
                              drop_last=True, **loader_kw)

    model = build_model(cfg).to(device)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(tr.get("ema_decay", 0.999))) \
        if tr.get("ema_decay") else None
    loss_fn = MultiHeadLoss(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    accum = tr.get("grad_accum", 1)
    scheduler = make_scheduler(optimizer, tr, max(1, len(train_loader) // accum))
    use_amp = tr.get("amp", True) and device == "cuda"

    best, start_epoch = -1.0, 1
    last_path = os.path.join(out_dir, "last.pt")
    if os.path.exists(last_path) and not fresh:
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        if ema is not None and ck.get("ema_full"):
            ema.load_state_dict(ck["ema_full"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        print(f"[auto-resume] epoch {start_epoch}, best PQ {best:.4f} (pass --fresh to restart)")

    log_path = os.path.join(out_dir, "metrics.csv")
    cols = ["epoch", "train_loss", "pq", "sq", "rq", "tp", "fp", "fn",
            "gt_split_frac", "pred_merge_frac", "lr", "minutes"]
    if start_epoch == 1:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(cols)

    pp = cfg["postprocess"]
    eval_size = cfg["train"].get("val_eval_size", 1024)
    t0 = time.time()
    for epoch in range(start_epoch, tr["epochs"] + 1):
        model.train()
        running, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"fold {fold} epoch {epoch}/{tr['epochs']}")
        optimizer.zero_grad(set_to_none=True)
        throttle = GpuThrottle()
        for step, (x, y) in enumerate(pbar, 1):
            throttle.begin()
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(x)
            loss, _ = loss_fn(logits, y)
            if not torch.isfinite(loss):  # never let one bad batch poison the weights
                optimizer.zero_grad(set_to_none=True)
                continue
            (loss / accum).backward()
            if step % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tr.get("grad_clip", 1.0))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                if ema is not None:
                    ema.update_parameters(model)
            running += loss.item()  # .item() synchronises, so the throttle measures real GPU time
            n += 1
            throttle.step(sync=False)
            pbar.set_postfix(loss=f"{running / n:.4f}")

        eval_model = ema.module if ema is not None else model
        is_last = epoch == tr["epochs"]
        if epoch % tr.get("val_every", 1) == 0 or is_last:
            probs = predict_fold(eval_model, val_obs, cfg, device)
            m = score_obs(obs, probs, disks, pp, eval_size)
            print(f"  val PQ={m['pq']:.4f} (SQ {m['sq']:.3f} x RQ {m['rq']:.3f}) "
                  f"TP {m['tp']} FP {m['fp']} FN {m['fn']} | split {m['gt_split_frac']:.2f} "
                  f"merge {m['pred_merge_frac']:.2f}")
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, f"{running / max(1, n):.5f}"] +
                                       [f"{m[k]:.5f}" if isinstance(m[k], float) else m[k]
                                        for k in cols[2:10]] +
                                       [f"{optimizer.param_groups[0]['lr']:.2e}",
                                        f"{(time.time() - t0) / 60:.1f}"])
            if m["pq"] > best:
                best = m["pq"]
                torch.save({"model": eval_model.state_dict(), "cfg": cfg, "epoch": epoch,
                            "pq": best}, os.path.join(out_dir, "best.pt"))
                print(f"  new best PQ {best:.4f} -> best.pt")

        if math.isfinite(running):
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "ema_full": ema.state_dict() if ema is not None else None,
                        "cfg": cfg, "epoch": epoch, "best": best}, last_path)

    # Out-of-fold probabilities from the best checkpoint, for post-processing tuning.
    ck = torch.load(os.path.join(out_dir, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    probs = predict_fold(model, val_obs, cfg, device)
    for oid, p in probs.items():
        save_probs(os.path.join(out_dir, "oof"), oid, p)
    print(f"[fold {fold}] best PQ {best:.4f}; OOF maps -> {out_dir}/oof")
    return {"fold": fold, "pq": best}


def main():
    limit_resources()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--fold", default="0", help="fold index or 'all'")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key=value")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    folds = range(cfg["data"].get("n_folds", 5)) if args.fold == "all" else [int(args.fold)]
    res = [train_fold(cfg, f, args.fresh) for f in folds]
    for r in res:
        print(f"fold {r['fold']}: PQ {r['pq']:.4f}")


if __name__ == "__main__":
    main()
