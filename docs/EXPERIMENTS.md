# Experiment log

All OOF numbers are pooled over 5 time-grouped folds (707 frames, 1,154 annotation sets),
scored with the organizers' PQ. Unless noted, scoring is at 1024 px; 1024 and 2048 agree within 0.002.
LB = public leaderboard (~90 test frames, ±0.03 noise).

| id | date | change | OOF PQ | SQ | RQ | LB | notes |
|---|---|---|---|---|---|---|---|
| B0 | 10-15 | baseline.yaml: U-Net R34, full disk @1024, soft targets, mask+boundary+spine heads, 40 ep | 0.32–0.35 per fold (default pp) | | | | best epoch ~26–30; FP grows afterwards |
| B0-pp | 10-16 | + tuned post-processing (thr 0.55, merge 10, min_area 600 native) | 0.363 | 0.648 | 0.560 | **0.31** | submitted pp was mis-scaled (pixel params tuned @1024, applied @2048); worth only −0.004 on OOF |
| B0-fix | 10-16 | post-processing pixel params in native units (resolution-independent) | 0.369 (fold 0) | 0.655 | 0.566 | | 1024 vs 2048 eval: 0.3690 vs 0.3709 |

## Error analysis of B0-pp (`analyze_oof.py`)

GT filaments (8,199):

| fate | share | meaning |
|---|---|---|
| matched (TP) | 51% | best IoU > 0.5 |
| unseen | 26% | no prediction overlaps it; half had confident pixels that the area filter removed |
| near miss | 12% | IoU 0.3–0.5: found, extent wrong |
| merged | 6% | its prediction also covers another GT |
| split | 3% | ≥2 predictions, none > 0.5 (mostly large filaments > 3k px: 14%) |

- **Size is the main driver.** TP rate is 0% under 300 px, 27% at 300–700, 58% at 700–1.5k and ~65% above.
- **Annotator agreement sets a ceiling.** TP rate is 11% on filaments only 1 annotator drew, but 69% on those all annotators drew.
- **False positives:** 30% match a filament another annotator drew (real, just unlabelled by this one), and 75% partially overlap some GT.
- **Instance mean probability is a poor confidence score:** precision is non-monotonic in it. This motivates the learned scorer.

## Learned selection (`train_scorer.py`, `src/candidates.py`)

The candidate tree uses levels 0.1–0.9, merge 10 px and min area 100 px, giving ~73 candidates per frame.
The scorer is a HistGBR cross-validated on the same folds; its target is the mean over annotators of IoU·[IoU>0.5].

| id | change | OOF PQ | Δ vs B0-pp | scorer corr |
|---|---|---|---|---|
| S1 | 28 features (prob stats, shape, disk position, contrast), levels 0.2–0.9 | 0.3709 (τ 0.16) | +0.008 | 0.512 |
| S1b | + levels 0.1, 0.15 | 0.3712 | +0.009 | 0.508 |
| S2 | + boundary sharpness, multi-radius contrast, tree context (40 features) | **0.3739** (τ 0.18) | **+0.011** | 0.535 |

S2 on the public LB: **0.32** (B0-pp: 0.31).

**Oracle ceiling:** selecting this same candidate set with the true targets gives **PQ 0.49**.
The scorer, not candidate generation, is the bottleneck. The best τ (0.16–0.18) is close to the theoretical PQ/2.

Test submission `runs/baseline/submission_scorer.csv` (5 folds, D4 TTA, S2 scorer):
1,088 filaments on 177/180 frames, 5.4 s/frame on a GPU shared with training.
Caveat: the scorer is trained on single-model OOF probabilities but applied to a 5-fold ensemble.
Ensembles are smoother, so this is a calibration shift.

Scorer GBM settings (S2 features, `--reuse-cache`):

| settings | OOF PQ |
|---|---|
| 31 leaves, min leaf 40 (S2) | 0.3739 |
| 1000 iter, lr 0.02 | 0.3743 |
| 15 leaves, min leaf 100 | 0.3765 |
| **15 leaves, min leaf 100, L2 5, 50% features per split** (new default) | **0.3778** |

More regularisation helps: the target is noisy (annotator disagreement).

## Fold-0 comparison (each tuned on fold 0 alone, so slightly optimistic)

| model | PQ | SQ | RQ | TP | FP | FN |
|---|---|---|---|---|---|---|
| B0 (1024 full disk) | 0.3819 | 0.657 | 0.582 | 795 | 423 | 721 |
| **C1 (2048, 512 crops)** | **0.4047** | **0.669** | 0.605 | 975 | 733 | 541 |
| B0 + C1, equal weights | 0.4051 | 0.659 | 0.615 | 923 | 562 | 593 |
| **B0 + C1, 0.25 / 0.75** | **0.4183** | **0.672** | **0.622** | 890 | 455 | 626 |

Native resolution is worth +0.023. Equal averaging with B0 adds nothing because B0's blurrier maps cost SQ,
but a little B0 (25%) helps: +0.014 over C1 alone, with far fewer FPs.

## Crop refiner (`train_refiner.py`, `src/refiner.py`) on B0, fold 0, exact PQ at 2048

The refiner is a U-Net R18 on 256 px crops. Its inputs are the image, the stage-1 probability and the candidate seed.
It outputs this filament's consensus mask plus a score. Training: folds 1-4 candidates, 8 x 16k samples (~40 min while sharing the GPU).

| variant | PQ | SQ | RQ |
|---|---|---|---|
| GBM selection, candidate masks | 0.3930 | 0.656 | 0.599 |
| **+ refined masks** | **0.4167** | 0.672 | 0.620 |
| blend GBM/refiner score 50/50 + refined masks | 0.4174 | 0.675 | 0.618 |

Refined outlines are worth **+0.024**: near misses become hits, so RQ rises along with SQ.
The refiner's score head is about as good as the GBM (corr 0.569 vs 0.573), and blending the two adds little (+0.0007).
`infer.py` uses `<run>/refiner/fold*.pt` automatically (`--no-refiner` to skip).

## Running
- C1: crop2048.yaml (native-res 512 crops). Fold 0 reached val PQ 0.360 by epoch 8 with untuned defaults,
  at much higher recall than B0 (TP 937 vs 731).

## Next
1. When C1 finishes, build the full stack on all folds:
   - ensemble 0.25 B0 + 0.75 C1 (`make_ensemble.py`)
   - scorer on the ensemble OOF
   - refiners for all 5 folds
2. Calibration: the scorer and refiner are trained on single-model OOF but applied to fold-ensemble probabilities.
3. Refiner: rotated crops along the filament axis, a neighbour-mask channel, a larger encoder.

## Full 5-fold stack on ENS1 = 0.75 C1 + 0.25 B0 (maps averaged at 1024)

| step | OOF PQ (all folds) | notes |
|---|---|---|
| B0, tuned threshold pp | 0.3627 | for reference |
| C1 alone, tuned threshold pp | 0.3934 | SQ 0.673, RQ 0.585, FP 4238 |
| **C2 (ConvNeXt) alone, tuned threshold pp** | **0.4245** | SQ 0.678, RQ 0.626, FP 3722: alone it matches the whole ENS1 stack |
| ENS1, tuned threshold pp | 0.4082 | SQ 0.671, RQ 0.608, FP 3076 |
| scorer on ENS1 | 0.4172 (tau 0.22) | +0.039 over the B0 scorer (0.3778); corr 0.615; oracle 0.551; 100 candidates/frame |
| + refiners (5 folds), exact 2048 scoring, GBM-only score | 0.4183 (tau 0.22) | refined masks |
| **+ score blend 0.3 GBM / 0.7 refiner** | **0.4244** (tau 0.22) | SQ 0.672, RQ 0.632, TP 4994, FP 2616, FN 3202; `tune_stack.py` |

**ENS1 stack on the public LB: 0.38** (`runs/ens1/submission_stack.csv`, 10-21), up from 0.32.
The OOF→LB gap is ~0.045, the same as before (0.363→0.31, 0.378→0.32), so OOF keeps ranking changes correctly.

The tau/blend surface is flat near the optimum (tau 0.22–0.24, blend 0.3–0.5 all within 0.0005), so the setting is robust.
With refiners in the loop, the refiner's image-based score is worth more than the GBM's.

## Ensemble weights on D4-TTA OOF maps (all 5 folds, tuned threshold pp)

`predict_oof.py --tta d4` regenerated the OOF maps of C2, C1 and B0 with the test-time TTA.

| blend (C2 / C1 / B0) | PQ | SQ | RQ | FP |
|---|---|---|---|---|
| 0 / 0.75 / 0.25 (= ENS1, D4 maps) | 0.4107 | 0.671 | 0.612 | 3031 |
| 0.4 / 0.4 / 0.2 | 0.4297 | 0.672 | 0.639 | 2611 |
| 0.45 / 0.3 / 0.25 | 0.4302 | 0.673 | 0.639 | 2409 |
| 0.5 / 0.3 / 0.2 | 0.4353 | 0.675 | 0.645 | 2669 |
| **0.6 / 0.2 / 0.2** | **0.4393** | 0.676 | 0.650 | 2596 |

| 0.6 / 0.15 / 0.25 | 0.4379 | 0.676 | 0.648 | 2460 |
| 0.65 / 0.2 / 0.15 | 0.4362 | 0.676 | 0.645 | 2970 |
| 0.8 / 0 / 0.2 | 0.4397 | 0.679 | 0.648 | 2792 |
| 0.75 / 0 / 0.25 | 0.4404 | 0.677 | 0.650 | 2566 |
| **0.7 / 0.1 / 0.2** (ENS2) | **0.4414** | 0.676 | 0.653 | 2562 |

D4 TTA on OOF maps: ENS1 0.4082 (hflip) → 0.4107 (D4), +0.0025.
The optimum is broad: everything with C2 0.6–0.8 and B0 0.2–0.25 lies within 0.002.
B0 (global context) at ~0.2 matters more than C1.

## Scorer on ENS2 (0.7 C2 / 0.1 C1 / 0.2 B0)

| scorer | OOF PQ | corr | oracle |
|---|---|---|---|
| tuned threshold pp (reference) | 0.4414 | | |
| GBM, 40 features | **0.4457** (tau 0.24) | 0.624 | 0.575 |
| + per-member mean/max probability + member spread (48 features) | 0.4448 (tau 0.26) | 0.620 | 0.575 |

Member-disagreement features bring nothing (within noise), so they are off by default (`member_features: true` in ensemble.yaml re-enables them).
With better stage-1 maps the scorer gains less over thresholds (+0.004 vs +0.015 on B0).
The oracle gap (0.446 vs 0.575) is now the main headroom; the refiner's image-based score is the lever.
Also: `resize_probs` now uses area interpolation when downsampling, matching `make_ensemble.py`.

## Final stack on ENS2 = W712 (0.7 C2 / 0.1 C1 / 0.2 B0), `tune_stack.py`, exact 2048 scoring

The refiners were trained on W622 (same members, weights 0.6/0.2/0.2). Candidate tree: merge 0 px. GBM regression scorer.

| tau \ GBM weight | 0.0 | 0.15 | 0.3 | 0.5 | 1.0 |
|---|---|---|---|---|---|
| 0.22 | 0.4421 | 0.4445 | 0.4452 | 0.4458 | 0.4432 |
| 0.24 | 0.4435 | 0.4462 | 0.4481 | 0.4485 | 0.4453 |
| 0.26 | 0.4456 | 0.4472 | 0.4496 | **0.4501** | 0.4455 |
| 0.28 | 0.4467 | 0.4489 | 0.4497 | 0.4496 | 0.4449 |

**OOF PQ 0.4501** (SQ 0.676, RQ 0.666, TP 5263, FP 2350, FN 2934), +0.026 over the ENS1 stack (0.4244, LB 0.38).
The surface is flat near the optimum.

**W712 stack on the public LB: 0.39** (v4, `runs/w712/submission_w712_stack.csv`, 10-22), up from 0.38.
OOF +0.026 became LB +0.01. The public score is rounded to 2 decimals and ~90 frames carry ±0.03 noise,
so this is consistent, but the OOF→LB gap is now ~0.06.
Inference under the compute caps took 85 s/frame (4.3 h for 180 frames), so speed has to be addressed before the final.

## Refiners on W622 (0.6 C2 / 0.2 C1 / 0.2 B0), 5 folds, exact 2048 scoring (`train_refiner.py` report)

| variant (best tau of 0.14/0.18/0.22) | mean PQ |
|---|---|
| GBM selection, candidate masks | 0.4341 |
| + refined masks | 0.4394 |
| **+ 50/50 GBM/refiner score blend** | **0.4444** |

The ENS1 stack that scored 0.38 on the LB was 0.4247 on the same measure. Optimal tau is still at the top of this grid, so `tune_stack.py` extends it.

## Error analysis of ENS2 (tuned threshold pp, `analyze_oof.py`), vs B0

| GT fate | B0 | ENS2 |
|---|---|---|
| matched | 51% | **64%** |
| unseen | 26% | 19% (65% of those had confident pixels dropped by filters) |
| near miss | 12% | 7.6% |
| merged | 6.2% | 5.7% |
| split | 3.2% | 3.1% (13% of filaments > 3k px) |

Matched rate by size: <300 px 0 → 11%, 300–700 px 27 → 51%.
By agreement: filaments drawn by 1 annotator of n are matched 18%; filaments drawn by all are matched 81%.
Merges barely moved, and each costs 2 FN + 1 FP. `boundary_cut` and spine-seeded grouping were never selected by tuning.
Candidate-tree variants on ENS2 (GBM scorer, all folds):

| tree | cands/frame | oracle | learned PQ | corr |
|---|---|---|---|---|
| merge 10 px, 10 levels (old) | 99 | 0.5753 | 0.4457 | 0.624 |
| **merge 0 px, 10 levels** (new default) | 107 | 0.5797 | **0.4470** | 0.648 |
| merge 20 px | 93 | 0.5640 | 0.4361 | 0.625 |
| merge 10 px, 18 levels | 172 | 0.5805 | 0.4454 | 0.631 |

Grouping fragments within 20 px hurts, so fragment grouping caused part of the merges. Pure connected components is best.

Scorer objective (merge-0 tree): regression of mean IoU·[IoU>0.5] **0.4470** (corr 0.648) vs P(match)×E[IoU|match] (`--objective hitiou`) 0.4462 (corr 0.652).
No gain; regression stays.
Doubling the levels adds nothing.

## B1: ConvNeXt-T full disk at 1024 (`configs/full1024_convnext.yaml`), fold 0

50 epochs, batch 2 x accum 4, D4-TTA OOF maps. Fold 0, tuned on fold 0:

| model | PQ | SQ | RQ |
|---|---|---|---|
| B0 (R34 full disk) alone | 0.3859 | 0.656 | 0.588 |
| **B1 (ConvNeXt full disk) alone** | **0.3982** | 0.651 | 0.611 |
| 0.6 C2 / 0.2 C1 / 0.2 B0 | 0.4360 | 0.680 | 0.642 |
| **0.6 C2 / 0.2 C1 / 0.2 B1** | **0.4397** | 0.680 | 0.647 |
| 0.6 C2 / 0.2 C1 / 0.1 B0 / 0.1 B1 | 0.4388 | 0.676 | 0.649 |

B1 is better than B0 alone (+0.012), but the blend gains only +0.004 on fold 0.

**All 5 folds** (D4 OOF maps, tuned threshold pp). Best val PQ per fold: 0.369 / 0.355 / 0.390 / 0.368 / 0.390.

| blend | PQ | SQ | RQ | FP |
|---|---|---|---|---|
| 0.7 C2 / 0.1 C1 / 0.2 B0 (W712) | 0.4414 | 0.676 | 0.653 | 2562 |
| **0.7 C2 / 0.1 C1 / 0.2 B1 (W71_B1)** | **0.4449** | 0.681 | 0.654 | 2187 |
| 0.7 / 0.1 / 0.1 B0 / 0.1 B1 | 0.4419 | 0.677 | 0.653 | 2841 |
| 0.65 / 0.1 / 0.25 B1 | 0.4440 | 0.680 | 0.653 | 2168 |

B1 replaces B0: sharper outlines and 15% fewer FPs.

Stack on W71_B1 (borrowed W622 refiners): scorer 0.4509 (corr 0.660, oracle 0.580).
**Stack 0.4536** (blend 0.5, tau 0.30, at the edge of the grid), vs 0.4501 for W712.
Refiners retrained on W71_B1, tau grid extended to 0.36: **stack 0.4542** (blend 0.3, tau 0.26, SQ 0.678, RQ 0.670, FP 2340, FN 2885).
That is only +0.0006 over the borrowed W622 refiners, so refiners transfer well between closely related blends.

**v5 on the public LB: 0.39** (same as v4). OOF +0.004 is below what the 2-decimal public score can resolve.
Inference: 98.5 s/frame under the caps (~5 h for 180 frames).

| submission | OOF PQ | public LB |
|---|---|---|
| v1 B0 threshold | 0.363 | 0.31 |
| v2 B0 + GBM scorer | 0.378 | 0.32 |
| v3 ENS1 stack | 0.4244 | 0.38 |
| v4 W712 stack | 0.4501 | 0.39 |
| v5 W71_B1 stack | 0.4542 | 0.39 |
| v6 W71_B1 stack, R34 refiners | 0.4569 | 0.40 |

## Refiner v2: ResNet34, 10 ep x 24k samples (vs R18, 8 x 16k), on W71_B1, fold 0, exact 2048 scoring

`train_refiner.py --run runs/w71_b1 --folds 0 --encoder resnet34 --epochs 10 --samples-per-epoch 24000 --tag _r34`.
Same candidates as the R18 refiner. The run was resumed from its epoch-3 checkpoint after a laptop restart.

| variant (best tau of 0.14/0.18/0.22) | R18 (current) | R34 v2 | delta |
|---|---|---|---|
| gbm (no refiner) | 0.4389 | 0.4389 | 0 |
| gbm + refined masks | 0.4382 | 0.4410 | +0.0028 |
| blend 0.5 + masks | 0.4440 | 0.4452 | +0.0012 |
| **blend 0.0 (refiner score only) + masks** | 0.4428 | **0.4482** | **+0.0054** |

Best R34 vs best R18: **+0.0042** (above the 0.003 bar). Both the masks (+0.003 with the GBM score) and the score head improve
(score corr 0.650, same as the GBM; blend 0.663). The best is at tau 0.22, the edge of this grid; `tune_stack.py` searches wider.
Training folds 1-4 (`runs/jobs/refiner_r34_f1234`).

All 5 folds (best over variant x tau 0.14/0.18/0.22, per-fold report; not pooled):

| fold | R18 | R34 | delta |
|---|---|---|---|
| 0 | 0.4440 | 0.4482 | +0.0041 |
| 1 | 0.4452 | 0.4463 | +0.0011 |
| 2 | 0.4567 | 0.4571 | +0.0004 |
| 3 | 0.4485 | 0.4514 | +0.0029 |
| 4 | 0.4644 | 0.4666 | +0.0022 |
| mean | 0.4518 | 0.4539 | **+0.0022** |

R34 wins on every fold, but by about half the fold-0 margin. Stack: `runs/w71_b1_r34/` (copy of w71_b1 with the R34 refiners).

`tune_stack.py --run runs/w71_b1_r34` (pooled 5-fold OOF, exact 2048): **stack 0.4569** (blend 0.3, tau 0.30; SQ 0.681, RQ 0.671,
TP 5137, FP 1987, FN 3060) vs **0.4542** for v5 = **+0.0027**. Interior optimum with a flat top (0.4564-0.4569 for blend 0.15-0.5, tau 0.26-0.30).
FP drops from 2340 to 1987. Refiner score corr 0.645-0.690 per fold, equal to or above the GBM's on 4 of 5 folds.

**v6 on the public LB: 0.40** (v5: 0.39). 1,070 filaments on 175/180 frames; inference 45 s/frame (D4) under the caps.

## C2: ConvNeXt-T U-Net, native 512 crops, 40 epochs (`configs/crop2048_convnext.yaml`)

Encoder weights are Meta's official ImageNet-22k→1k checkpoint (`dl.fbaipublicfiles.com`), loaded through timm's
`pretrained_cfg_overlay`, because Hugging Face is blocked on this network. Loading was verified: a deep layer matches
the checkpoint exactly and the stem was adapted to 1 channel. Fold 0: best val PQ 0.3999 at epoch 32 (default pp), ~2 min/epoch.

| fold 0, tuned on fold 0 | PQ | SQ | RQ | FP |
|---|---|---|---|---|
| C1 (ResNet34) | 0.4047 | 0.669 | 0.605 | 733 |
| **C2 (ConvNeXt-T)** | **0.4217** | 0.678 | 0.622 | 680 |
| C1 + C2 (0.5/0.5) | 0.4191 | 0.677 | 0.619 | 618 |
| C1 + B0 (0.75/0.25), the current ENS1 | 0.4183 | 0.672 | 0.622 | 455 |
| **C1 + C2 + B0 (0.4/0.4/0.2)** | **0.4323** | 0.675 | 0.641 | 446 |

C2 is the best single model (+0.017 over C1). The 1024 full-disk B0 still adds a lot in a blend: it removes FPs
thanks to its global context. Training C2 folds 1–4.

An EfficientNet-B4 variant (`configs/crop2048_effb4.yaml`) hung at its first batch while starting dataloader workers
(GPU idle, workers idle). It did not reproduce in a direct probe and is parked; py-spy is installed to dump stacks if it recurs.

## Inference speed (`tools/profile_infer.py`, W71_B1, per frame, under the compute caps)

Two bugs, not the models, made inference slow:
1. The GPU duty throttle timed from its previous call, so unrelated CPU work and other models counted as busy time. It now times only the GPU work (`begin()`/`step()`).
2. Refiner batch sizes changed on every frame (the candidate count varies). With `cudnn.benchmark`, every new shape re-runs kernel autotuning (~25 s/frame). Batches are now padded to 64.

| stage | D4 (deployed) | hflip |
|---|---|---|
| C2 ConvNeXt crops x5 | 20.8 s | 5.4 s |
| C1 R34 crops x5 | 14.1 s | 3.6 s |
| B1 ConvNeXt full disk x5 | 3.5 s | 0.9 s |
| refiners x5 (~74 candidates) | 3.6 s (was 29 s) | |
| preprocess + candidates + GBM + paint | 0.5 s | |
| **total** | **42 s** (was 98 s) | ~14 s |

D4 vs hflip OOF maps: +0.0025 PQ (ENS1). D4 is kept for accuracy; dropping it is the next lever if speed is needed.

## Target pixel convention (10-27, `tools/check_target_geometry.py`, `tests/test_targets.py`)

COCO polygon vertices lie on pixel edges, while cv2 `fillPoly` and `resize` work in pixel centres. Our cv2 targets
therefore sat +0.5 px off the organizers' pycocotools masks. The refiner check: 300 matched candidates, consensus target
pasted back to 2048 and compared with the consensus of the `polygon_to_rle` GT (>0.5):

| refiner target geometry | median IoU | p10 | mean |
|---|---|---|---|
| square (as trained for v6) | 0.889 | 0.780 | 0.878 |
| square, −0.5 crop-px fix | 0.919 | 0.806 | 0.900 |
| **rot (384x160, already used poly − 0.5)** | **0.934** | **0.832** | **0.915** |

The rotated crop is not worse than square, so the rot refiner is trained (fold-0 gate). The square-crop target fix is now the
default in `train_refiner.py` (`refiner_sqfix_f0` gate; v6 refiners were trained without it).
Stage-1 targets (`src/targets.py`) have the same offset. On synthetic thin quads, target vs pycocotools IoU is 0.916 → 0.939 at
2048 and 0.838 → 0.889 at 1024 with `data.target_offset: -0.5`. It is opt-in (default 0.0 = every existing model);
gate `c2tfix_f0` is C2 fold 0 retrained with it.

## Machine move to the RTX 4090 desktop (10-27)
i9-14900KF (32 threads), 64 GB RAM, RTX 4090 24 GB (driver 560.94; the torch 2.11 cu128 wheel works on sm_89).
Caps on this dedicated machine: `SF_CPU_FRAC 0.75`, `SF_GPU_MEM_FRAC 0.9`, `SF_GPU_DUTY 1.0` (user env vars; `src/resources.py`
falls back to `HKCU\Environment` because WMI-launched jobs may not inherit them, and logs the active caps).
C2 stage-1 training (crop2048_convnext): ~1.0 min/epoch incl. validation, vs ~2 min (early) to ~35 min (I/O-throttled) on the laptop.

## C2 seed 2 (`seed=1042`, `runs/crop2048_convnext_s2`, done 10-27 16:08)
Best val PQ per fold (default pp, hflip), seed 1 → seed 2: 0.3999 → 0.3978, 0.3925 → 0.3905, 0.4113 → 0.4041, 0.4026 → 0.4004, 0.4220 → 0.4236.
Mean 0.4057 → 0.4033: seed-to-seed spread per fold is ~0.002–0.007, a useful noise bar for fold-0 gates.
D4 OOF maps: `runs/crop2048_convnext_s2_d4` (707 frames in 8 min on the 4090). Ensemble value still to be measured.
Refiner training on the 4090: ~3 min/epoch (10 x 24k) ≈ 35 min/fold, CPU-bound (4 loader workers, GPU mostly idle).

## Fold-0 gates of 10-27 (4090)
Refiners (R34, 10 ep x 24k, W71_B1 candidates; best over variant x tau 0.14/0.18/0.22, same measure as the v6 R34 report):

| refiner | best PQ | SQ | RQ | FP | FN |
|---|---|---|---|---|---|
| square (v6) | 0.4482 (blend 0.0, tau 0.22) | 0.678 | 0.661 | 513 | 514 |
| square, −0.5 px target fix | 0.4477 (blend 0.5, tau 0.22) | 0.679 | 0.660 | 496 | 526 |
| **rot 384x160** | **0.4532** (blend 0.5, tau 0.22) | **0.684** | 0.662 | 484 | 526 |

The rotated crop gains **+0.0051**, mostly through SQ (+0.006): outlines are sharper with ~1.2x more pixels along the filament.
The square-target fix alone is neutral (−0.0004) even though its targets match the GT better (0.889 → 0.919).
The pixel offset matters less than resolution, probably because tau and blend re-tune around it. Rot goes to folds 1-4.

Stage-1, fold 0 alone, D4 OOF maps, `tune_postprocess` on fold 0 (`runs/f0_*`):

| model | PQ | SQ | RQ | FP | best val PQ (default pp) |
|---|---|---|---|---|---|
| C2 ConvNeXt-T (reference) | **0.4314** | 0.680 | 0.634 | 652 | 0.3999 @32 |
| C3S ConvNeXt-S | 0.4290 | 0.678 | 0.633 | 613 | 0.4000 @24 (40 min) |
| C2 + `data.target_offset -0.5` | 0.4307 | 0.680 | 0.633 | 587 | 0.4025 @28 |

ConvNeXt-S is not better than ConvNeXt-T on its own. Its earlier peak (epoch 24) hints at overfitting with this data size.
The stage-1 target offset is within seed noise. Neither is adopted for now; ConvNeXt-S may still help as an extra ensemble member.

## v7: W71_B1 stack with rotated-crop R34 refiners (10-28, `runs/w71_b1_rot`)
Rot refiners on all 5 folds (10 ep x 24k), `tune_stack.py`: **stack OOF 0.4623** (blend 0.5, tau 0.26; SQ 0.685, RQ 0.675,
TP 5369, FP 2335, FN 2828), +0.0054 over v6 (0.4569). Test inference on the 4090: **6.5 s/frame** with D4 (hflip: 3.2 s/frame).
**v7 public LB 0.40** (v6 0.40): the gain is below the 2-decimal resolution.

Test-set probes of v7 (same cached stage-1 maps):

| probe | filaments | public LB |
|---|---|---|
| v7 (refiner, tau 0.26) | 1168 | 0.40 |
| refiner off (GBM scorer only) | 1154 | 0.39 |
| tau 0.38 | 907 | 0.39 |
| tau 0.22 | 1263 | 0.39 |
| hflip TTA instead of D4 (10-29; 3.2 vs 6.5 s/frame) | 1178 | 0.40 |

Both tau directions and refiner-off lose on the test set too (−0.01 at 2-decimal resolution), so the OOF optimum transfers. There is
no detectable calibration shift between single-model OOF maps and the 5-fold test ensemble, and the refiner is confirmed on test.

## C2 seed 2 as an ensemble member (`ens_s2`, D4 OOF maps, tuned threshold pp, all folds)

| blend | PQ | SQ | RQ | FP |
|---|---|---|---|---|
| 0.7 C2 / 0.1 C1 / 0.2 B1 (W71_B1) | 0.4449 | 0.681 | 0.654 | 2187 |
| 0.35 C2 / 0.35 C2s2 / 0.1 C1 / 0.2 B1 | 0.4480 | 0.680 | 0.658 | 2363 |
| **0.4 C2 / 0.4 C2s2 / 0.2 B1** (`w_s2_44`) | **0.4492** | 0.680 | 0.661 | 2473 |
| 0.4 / 0.4 / 0.05 C1 / 0.15 B1 | 0.4493 | 0.681 | 0.660 | 2480 |

A second C2 seed gains +0.0044 and makes C1 redundant (`w_s2_44` drops it, so inference is cheaper). The stack for it is v8.

## Annotation protocol (MAGFiLO paper, Ahmadzadeh et al. 2024, Sci Data 11:1031) checked on train labels
The protocol says filaments beyond ~±70° from the central meridian were not annotated, masks are one piece without holes and tightly fit
the body including barbs, small grouped filaments and too-faint ones were skipped, and splitting a fragmented filament is annotator-dependent.
Train GT confirms the longitude rule: |lon| > 65° holds 0.5% of 8,199 filaments, > 70° 0.05%, max 74°. Our W71_B1 candidates beyond
65° (835, 1.1%) have mean target y 0.025 vs 0.32 inside. The scorer only has the radial `r_disk`, which cannot tell the east/west limb
(not annotated) from the poles (annotated). Next: add |longitude| / |latitude| features.
Result (`train_scorer --tag _lon` on w_s2_44): 0.4529 vs 0.4525, i.e. +0.0004. The GBM already handled it through r_disk and the other
features, and only ~1% of candidates are affected. The features stay in `FEATURES` (neutral; new scorers include them).

## BiCC loss gate (C2 fold 0, `loss.heads.mask.bicc=0.3`, `runs/f0_C2B`)
Tuned fold-0 PQ **0.4317** vs 0.4314 for C2: no gain. The untuned val PQ rose (0.4144 vs 0.3999 at default pp), so BiCC mainly shifts
calibration, and threshold tuning recovers the same operating point. Not adopted.

## v8: w_s2_44 stack (0.4 C2 / 0.4 C2 seed 2 / 0.2 B1), borrowed W71_B1 rot refiners (10-28, `runs/w_s2_44`)
Scorer 0.4525 (corr 0.661, oracle 0.5845). `tune_stack`: **stack OOF 0.4647** (blend 0.5, tau 0.28; SQ 0.685, RQ 0.678, TP 5329,
FP 2195, FN 2867), +0.0024 over v7. Inference 4.1 s/frame (C1 dropped). **v8 public LB 0.40.**

## Overnight 10-28/29: more stage-1 members (4090, ~40 min/fold crop, ~20 min/fold full disk)
Best val PQ per fold (default pp):

| run | f0 | f1 | f2 | f3 | f4 |
|---|---|---|---|---|---|
| C2 seed 1 (reference) | 0.3999 | 0.3925 | 0.4113 | 0.4026 | 0.4220 |
| C2 seed 3 (`crop2048_convnext_s3`) | 0.4036 | 0.3930 | 0.4070 | 0.3992 | 0.4232 |
| C3S ConvNeXt-S (`crop2048_convnext_small`) | 0.4000 | 0.3883 | 0.4110 | 0.4009 | 0.4216 |
| B1 seed 2 (`full1024_convnext_s2`) | 0.3700 | 0.3662 | 0.3837 | 0.3899 | 0.3952 |

Blends (`blends_v9`) and fold-0 gates (`gates_0429`: R50 rot refiner, ConvNeXt-B, crop 768) are running.

Blends (D4 OOF maps, tuned threshold pp, all folds; reference `w_s2_44` 0.4492):

| blend | PQ | FP |
|---|---|---|
| `w9_3seed` 0.27 C2 / 0.27 s2 / 0.26 s3 / 0.2 B1 | 0.4496 | 2468 |
| `w9_2seed_cs` 0.27 C2 / 0.27 s2 / 0.26 C3S / 0.2 B1 | 0.4494 | 2473 |
| `w9_3seed_2b1` 0.27 / 0.27 / 0.26 s3 / 0.1 B1 / 0.1 B1s2 | 0.4503 | 2379 |
| **`w9_all`** 0.2 C2 / s2 / s3 / C3S, 0.1 B1 / B1s2 | **0.4511** | 2351 |

Returns from more 512-crop seeds are small now (+0.0004 for a third seed, +0.0019 for everything).

Fold-0 gates (`gates_0429`):

| gate | fold-0 PQ | reference | delta |
|---|---|---|---|
| R50 rot refiner (`refiner_rot_r50`) | 0.4527 | R34 rot 0.4532 | −0.0005: no gain |
| ConvNeXt-B stage-1 (`f0_C4B`) | 0.4339 | C2 0.4314 | +0.0025 (within seed noise) |
| **C2 with 768 crops** (`f0_C768`, `--set data.crop_size=768`) | **0.4442** (SQ 0.678, RQ 0.655, FP 466) | C2 0.4314 (FP 652) | **+0.0128** |

More context per crop is the biggest stage-1 gain since C2 itself: 30% fewer FPs at the same SQ. Training folds 1-4 (`c768_f1234`,
~47 min/fold), then a 1024-crop gate (`c1024_f0`).

C768 on all folds (best val PQ per fold, default pp), vs C2 at 512: 0.4145 / 0.4046 / 0.4183 / 0.4131 / 0.4395 vs
0.3999 / 0.3925 / 0.4113 / 0.4026 / 0.4220: **+0.012 on average, better on every fold**.
C768 seed 2: 0.4130 / 0.4007 / 0.4213 / 0.4127 / 0.4322. 1024 crops, fold 0: 0.4433 (tuned) vs 768 0.4442: no further gain, so 768 it is.

Blends with C768 (10-30, tuned threshold pp, all folds; references `w_s2_44` 0.4492, `w9_all` 0.4511):

| blend | PQ | FP |
|---|---|---|
| `w10_768x2_b1` 0.4 C768 / 0.4 C768s2 / 0.2 B1 | 0.4482 | 2432 |
| `w10_768x2_b1x2` 0.4 / 0.4 / 0.1 B1 / 0.1 B1s2 | 0.4489 | 2357 |
| `w10_768x2_c2_b1` 0.3 / 0.3 / 0.1 C2 / 0.1 C2s2 / 0.2 B1 | 0.4500 | 2386 |
| `w10_768x2_c2_b1x2` 0.3 / 0.3 / 0.1 / 0.1 / 0.1 / 0.1 | 0.4506 | 2318 |

**The fold-0 gain of 768 crops (+0.013 alone) does not survive blending**: inside the ensemble C768 is worth about what C2 is.
Probably the full-disk B1 member already supplies the context that removes FPs, which was C768's advantage.
All 5-fold blends since v8 sit at 0.449–0.451: the threshold-pp ensemble has plateaued. Checking C768 alone on all folds (`f_C768` vs `f_C2`).
ConvNeXt-B at 768 crops, fold 0: 0.4503 (vs C768 0.4442, +0.006); folds 1-4 training.

Single models on all folds (tuned threshold pp, D4): **C768 0.4405** vs C2 0.4334 (+0.007; FP 3239 vs 3619). The 768-crop gain is real
on all folds (smaller than fold 0's +0.013), but it vanishes in blends. 8-member blend with C768 seed 3: 0.4508.
ConvNeXt-B 768 (C768B) all folds, best val per fold: 0.4186 / 0.3989 / 0.4238 / 0.4087 / 0.4391, mean 0.4180 = C768's 0.4180.

## v9: w9_all stack (10-30, `runs/w9_all`: 0.2 C2 / 0.2 C2s2 / 0.2 C2s3 / 0.2 C3S / 0.1 B1 / 0.1 B1s2), borrowed W71_B1 rot refiners
Scorer + `tune_stack`: **stack OOF 0.4654** (blend 0.5, tau 0.26; SQ 0.685, RQ 0.679, TP 5408, FP 2318, FN 2789), +0.0007 over v8.
**v9 public LB 0.40.**

## 11-01 gates and blends
| test | result | reference | verdict |
|---|---|---|---|
| v10: dedicated rot refiners trained on w9_all candidates | stack 0.4632 (tau 0.30) | v9 with borrowed W71_B1 rot refiners 0.4654 | −0.002, not uploaded |
| rot refiner 20 epochs (fold 0) | 0.4493 | 10 epochs 0.4532 | worse |
| C768 with 60 epochs (fold 0) | 0.4426 | 40 epochs 0.4442 | no gain |
| C768B (ConvNeXt-B, 768 crops) alone, all folds | **0.4445** | C768 0.4405, C2 0.4334 | best single model |
| `w11_all_768b` (0.13 C2 / s2 / s3 / C3S, 0.14 C768B / C768, 0.1 B1 / B1s2) | **0.4523** | w9_all 0.4511 | +0.0012 → v11 stack |
| `w11_768b_768_b1x2` | 0.4494 | | |

Refiners trained on a closely related ensemble transfer as well as dedicated ones (seen twice now: +0.0006 and −0.002); longer refiner
training overfits. Note that v10 overwrote `runs/w9_all/refiner/` and `stack.yaml`; the v9 files were restored (the dedicated
refiners and their sweep are kept in `runs/w9_all/refiner_rot/` and `runs/w9_all/ownrot_10ep/`).

## v11: w11_all_768b stack (11-02, borrowed W71_B1 rot refiners)
Scorer 0.4541 (corr 0.661). `tune_stack`: **stack OOF 0.4667** (blend 0.5, tau 0.28; SQ 0.686, RQ 0.680, FP 2084, FN 2899), +0.0013
over v9. **v11 public LB 0.40.**
C768B seed 2, best val per fold: 0.4192 / 0.4012 / 0.4216 / 0.4112 / 0.4410.
B1B (full disk 1024, ConvNeXt-B, 50 epochs, ~30 min/fold): 0.3790 / 0.3651 / 0.3930 / **0.3032** / 0.4087. Fold 3 never left
the all-background solution until epoch 42 (train loss 0.63 vs 0.40 for B1); retraining it with another seed.
Retrained fold 3 (`--fresh seed=1042`): 0.3746. **B1B per fold 0.379 / 0.365 / 0.393 / 0.375 / 0.409 vs B1 0.369 / 0.355 / 0.390 /
0.368 / 0.390: +0.010 on average.** C768B seed 3: 0.4190 / 0.4062 / 0.4279 / 0.4120 / 0.4386.
Blends: `w12_9m` (+ C768B s2) 0.4527 vs `w11_all_768b` 0.4523; `w12_768b2_heavy` 0.4518. The crop side has saturated;
B1B blends (`w13_*`) are next, built and uploaded unattended by `tools/auto_stack.py`.

## 11-02 evening: B1B blends, full disk at 1536
Blends (tuned threshold pp): `w13_b1all` (8 crop members + B1 / B1s2 / B1B) **0.4533**, `w13_10m_b1` 0.4529, `w13_b1b` (B1B only) 0.4518.
`auto_stack` built the `w13_b1all` stack: **0.4663** vs v11 0.4667, so not uploaded. B1B seed 2: 0.380 / 0.344 / 0.376 / 0.376 / 0.397.
**Full disk at 1536 (ConvNeXt-T, `--set data.input_size=1536`, ~42 min/fold):** best val per fold 0.3961 / 0.3807 / 0.4176 / 0.3950 /
0.4225 vs B1 at 1024 0.369 / 0.355 / 0.390 / 0.368 / 0.390: **+0.027 on every fold**, almost at the crop models' level. Resolution,
not encoder size, limits the global-context member. Blends `w14_*` (F15 weight 0.2-0.4) → `v12b_auto`.
Result: `w14_f15_b1s` (F15 0.1 + B1 0.05 + B1B 0.05) 0.4536, `w14_f15_20` 0.4527, `w14_f15_30` 0.4505, `w14_f15_40` 0.4477 → no stack built.
A much stronger single member (+0.027) adds nothing to the blend: **the threshold-pp ensemble has saturated at ~0.453** (it was
0.4492 at v8 and is within +0.004 after 8 more members). Remaining headroom is in stage 2 (oracle 0.58 vs stack 0.467).

Incident 11-02 23:36 → 11-03 08:50: `auto_stack.py` exited early (no gain) without waiting for its `--wait-job`, so the next GPU chain
started next to the running one; both ran at 530 s/batch with 23.8 GB used, for 9 h. Fixed (it now always waits); both jobs resumed
from their checkpoints.

## Level-2 selection on refined-mask targets (11-03, `tune_stack_l2.py`, `runs/w11_all_768b`)
Targets of the *refined* masks (mean over annotators of IoU·[IoU>0.5]) were computed for 58,282 of 73,097 candidates (mean 0.416 vs
0.395 for the candidate masks). A level-2 GBM on [candidate features, GBM score, refiner score, refined area, area ratio], trained out-of-fold:
corr with the refined target 0.608 (GBM 0.580, refiner 0.584). Selection PQ (best tau):

| score | PQ |
|---|---|
| tune_stack blend 0.5 (v11) | **0.4666** |
| 0.5 L2 + 0.5 blend | 0.4659 |
| 0.7 L2 + 0.3 blend | 0.4650 |
| L2 only | 0.4645 |

Better correlation, no better selection: around the keep threshold, ranking errors come from annotator disagreement that no score can
predict. Not adopted. The oracle (0.58) uses the true per-annotator targets and is not a reachable ceiling.
Next: average 2-3 independently seeded rot refiner sets (`train_refiner.py --seed`; `tune_stack.py` and infer average every
`fold{k}*.pt` in `refiner/`), queued as `v12_r2`.

Refiner seed ensemble (11-04): a second rot R34 refiner set (`--seed 1`, per-fold best 0.4541 / 0.4494 / 0.4635 / 0.4539 / 0.4727)
averaged with the first in `runs/w11_r2`: **stack 0.4658** vs v11 0.4667. No gain; seed 2 trained but not tested.
All stage-2 variants since v11 land at 0.465-0.467: the differences are noise (re-running v11's own blend gives 0.4666).

## Inference speed of candidate final pipelines (11-04, RTX 4090, uncached, 180 test frames)

| pipeline | stage-1 models | TTA | s/frame | OOF PQ | public LB |
|---|---|---|---|---|---|
| v11 (`w11_all_768b`) | 8 x 5 folds | D4 | 18.7 | 0.4667 | 0.40 |
| v11 | 8 x 5 folds | hflip | 6.7 | (D4 OOF maps) | 0.40 |
| v8 (`w_s2_44`) | 3 x 5 folds | hflip | **3.2** | 0.4647 (D4 OOF maps) | 0.40 |

On the public set all three tie at 0.40, while v8 + hflip is 5.8x faster than v11 + D4 for −0.002 OOF.
Honest OOF of the fast pipeline (11-06, `runs/w_s2_44h`: hflip OOF maps for C2 / C2s2 / B1, threshold tune 0.4463, scorer 0.4517,
borrowed rot refiners, `tune_stack` blend 0.5 tau 0.30): **stack OOF 0.4623** vs 0.4647 on D4 maps (−0.0024, as estimated on ENS1).
The fast final must run from `runs/w_s2_44h` (scorer and tau tuned on hflip maps). Its test run: 3.32 s/frame, **public LB 0.40**.

## Final A vs final B, paired on OOF (11-11, `runs/report/oof_v11.csv` vs `oof_fastB.csv`, 707 frames)
Per-frame TP-IoU sum / TP / FP / FN of both pipelines, resampled together:

| measure | A − B (PQ) |
|---|---|
| all frames | +0.0045 (0.4667 vs 0.4622) |
| bootstrap over frames, 95% interval | +0.0018 to +0.0072, P(A>B) 0.999 |
| cluster bootstrap over 214 14-day groups, 95% interval | +0.0016 to +0.0073, P(A>B) 0.999 |
| random 90-frame subsets (size of the private half) | sd 0.0036, P(A>B) 0.90, P(B ahead by ≥0.005) 0.004 |
| per fold | +0.0068 / +0.0002 / +0.0064 / +0.0051 / +0.0048 |

A is really better, not noise, but by less than the 2-decimal LB resolves; B almost never beats A by a visible margin.
If only one can be chosen for PQ it is A; B is the efficiency entry (5.7x faster). Caveat: A's blend was picked among more
candidate blends than B's, so a small part of the gap may be selection bias.
Laptop check (8 GB, new venv): B re-run on 3 test frames matches the submitted CSV (15/15 filaments, IoU ≥ 0.9985).
A cannot run on the laptop: its 40 checkpoints exceed the 5.6 GB GPU cap while loading (`infer.py` loads every member up front).

## Test-set ablation ladder (public LB, ~90 frames, 2 decimals; for the report)
All from the same cached D4 stage-1 test maps; threshold rows use each run's tuned threshold post-processing.

| pipeline | OOF PQ | public LB |
|---|---|---|
| C2 (ConvNeXt-T, 512 crops) alone, threshold | 0.4245 | 0.37 |
| W71_B1 ensemble (0.7 C2 / 0.1 C1 / 0.2 B1), threshold | 0.4449 | 0.38 |
| w_s2_44 ensemble (0.4 C2 / 0.4 C2s2 / 0.2 B1), threshold | 0.4492 | 0.39 |
| W71_B1 + GBM candidate scorer (no refiner) | 0.452* | 0.39 |
| W71_B1 + scorer + rot refiners (v7) | 0.4623 | 0.40 |
| w_s2_44 + scorer + rot refiners (v8) | 0.4647 | 0.40 |

*approximate (scorer-only OOF of the same stack). Every stage that gains on OOF also gains (or ties at 2-decimal resolution) on the
public test half, and the OOF → LB offset is a steady ~0.05–0.06.
CPU-only threshold inference crashed natively every ~70 frames (see HANDOFF gotchas); resumed from `.parts`.

## Fixes and incidents
- **10-16, GPU driver hang.** crop2048 crashed natively (c10 AbortHandler) on fold 2 / epoch 29 while test inference shared the GPU.
  The driver wedged: `nvidia-smi` and CUDA init hung until a reboot. Training resumed from `last.pt`.
  Rule from now on: no second GPU job while a training run is active.
- **10-20, white dot at the disk centre.** The 4x-subsampled limb profile left the innermost radial bins empty;
  they defaulted to 1.0 and the centre (~60 px) saturated to 255. Empty bins are now interpolated from their neighbours.
  Everything else is unchanged (mean |diff| 0.02 grey levels). Existing models were trained with the dot.
  Regenerate `data/cache` (`prepare_data.py`) before training new models.
- **10-21, second laptop hang** with 5 parallel tuning jobs plus GPU training. Every entry point now calls
  `src/resources.limit_resources()`: below-normal priority, 60% of CPU threads (torch/OpenMP/OpenCV/threadpoolctl, DataLoader workers),
  a 70% GPU memory cap, and a GPU duty-cycle throttle (~65%). Jobs run one at a time.
  Measured on refiner training: GPU ~63% average, total CPU ~42%.
- **10-22, Claude restart killed a running job** (refiner training, fold 0). Long jobs now start via `tools/detached.ps1`
  (WMI `Win32_Process.Create`, outside the caller's process tree). It writes `runs/jobs/<name>.{log,status}`.
- **10-22/23, lost ~11 h**: the launcher wrote its status with `echo DONE %errorlevel%> file`. On exit code 0, cmd parses `0>` as
  "redirect stream 0", so the file stayed empty and the queued inference job waited forever. Fixed (redirect-first syntax).
- **10-23, laptop restart** killed v5 inference at frame 56/180, since an OS restart ends even detached jobs.
  `infer.py` now writes a per-frame checkpoint (`<out>.parts/<obs>.csv`) and resumes from it.
  `tools/resume.ps1` relaunches every job still marked RUNNING after a restart.
- **10-23, file I/O throttling + 3 laptop restarts.** From ~15:55, reading one cached PNG took ~4 s (kernel "System" ~600% CPU,
  likely endpoint/inventory agents); refiner training dropped to ~60 s/batch. A restart at 18:03 killed the R34 refiner at 87% of its
  last epoch, losing the whole fold. `train_refiner.py` now saves `<fold>.pt.last` after every epoch and resumes from it;
  a finished `<fold>.pt` is reused. Verified: after the 18:44 restart it resumed at epoch 4.
  `.vscode/settings.json` excludes data/, runs/, weights/ and the venvs from VS Code watching/indexing.
- `visualize.py`: OOF overlays (annotator outlines; matched predictions green, unmatched red) for qualitative review and the report.
