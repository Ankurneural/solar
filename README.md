# SolarFragments — Solar Filament Segmentation Challenge 2026

Instance segmentation of solar filaments in GONG H-alpha full-disk frames
([Kaggle](https://www.kaggle.com/competitions/filament-segmentation-2026)).
Scored with **Panoptic Quality** (PQ), an instance-level metric, against every
annotator separately. Read [docs/COMPETITION_NOTES.md](docs/COMPETITION_NOTES.md)
first (metric, data quirks, state of the field, plan) and
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for every result and ablation.

## Method in one paragraph

Frames are limb-flattened and disk-masked. Stage-1 U-Nets predict soft, annotator-averaged
**mask / boundary / spine** maps:
- native-resolution models trained on 512 or 768 px crops of the 2048 px frame (ConvNeXt-T/S/B encoders)
- full-disk 1024 px models (ConvNeXt-T) for global context, which remove false positives

Their probabilities are averaged. Thresholding the mask at 10 levels gives a **tree of nested
candidate filaments**. Each candidate is scored for its expected contribution to PQ
(IoU · [IoU > 0.5], averaged over annotators) by two models:
- a gradient-boosted model on 42 hand-crafted features
- a **rotated-crop refiner** (ResNet34 U-Net on a crop aligned with the candidate's principal axis: image,
  probability and candidate mask) that also redraws the filament's outline

An exact dynamic program on the tree keeps the disjoint set of candidates whose blended
score exceeds τ ≈ PQ/2. That is the rule under which adding a filament raises PQ.

## Results

| | run | stage-1 models | TTA | OOF PQ | public LB | s/image (RTX 4090) |
|---|---|---|---|---|---|---|
| **A** accuracy | `runs/w11_all_768b` | 8 × 5 folds | D4 | 0.4667 | 0.40 | 18.7 |
| **B** efficiency | `runs/w_s2_44h` | 3 × 5 folds | hflip | 0.4623 | 0.40 | 3.3 |

**A is the final submission; B is its efficient variant** (also scored on the public leaderboard). A is ahead of B by +0.0045 OOF PQ (paired bootstrap over 14-day groups: 95% interval +0.0016 to +0.0073). One expert scored against the other experts reaches PQ 0.336. Every number, ablation and negative result is in
[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md); the report draft is in [docs/report/](docs/report/).

## Reproduce the submissions

- **Kaggle:** [notebooks/reproduce_kaggle.ipynb](notebooks/reproduce_kaggle.ipynb) (public Kaggle notebook
  [`ankurparmar/solarfragments-reproduce`](https://www.kaggle.com/code/ankurparmar/solarfragments-reproduce)) regenerates B end to end
  (about 2 h on a Kaggle T4; A needs about 24 h there, over the 12 h session limit, so run it locally on a GPU with at least 16 GB)
  from the public dataset [`ankurparmar/solarfragments-weights`](https://www.kaggle.com/datasets/ankurparmar/solarfragments-weights)
  (inference code, checkpoints, pinned wheels from [requirements-infer.txt](requirements-infer.txt));
  `tools/stage_kaggle_dataset.ps1` rebuilds that dataset folder from the repo.
- **Locally**, with the trained runs in `runs/`:
  ```bash
  python infer.py --run runs/w11_all_768b --tta d4 --out submission_A.csv
  python infer.py --run runs/w_s2_44h --tta hflip --out submission_B.csv
  python tools/submit.py submission_B.csv     # validate only (--upload to submit)
  ```
Outputs match the submitted files up to a few pixels per filament (IoU ≥ 0.997): cuDNN picks kernels per run.
On GPUs without bf16 (T4, P100) inference runs in fp16 with an fp32 fallback.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest tests -q
```

ImageNet-22k ConvNeXt weights (Meta's official checkpoints; Hugging Face is not needed) go in `weights/`:

```bash
curl -L -o weights/convnext_tiny_22k_1k_224.pth https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_224.pth
curl -L -o weights/convnext_small_22k_1k_224.pth https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_224.pth
curl -L -o weights/convnext_base_22k_1k_224.pth https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth
```

## Data

Join the competition and accept the rules on Kaggle, then download "MAGFiLO_1.0_Kaggle_2026"
from the Data tab and unzip it into `data/`:

```
data/MAGFiLO_1.0_Kaggle_2026/
  train/train_images/*.jpg
  train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
  test/test_images/*.jpg
```

Do not use the public MAGFiLO 1.0 release: it contains labels for the test images
and using it disqualifies a submission. Inference uses only the supplied test image.

## Pipeline

| step | command | output |
|---|---|---|
| preprocess + folds | `python prepare_data.py --config configs/baseline.yaml` | `data/cache/{images/,disks.csv,folds.csv,gt_train.csv}` |
| dataset stats, human ceiling | `python eda.py` | `data/cache/eda/` |
| stage 1, 5 folds each | `python train.py --config configs/<cfg>.yaml --fold all` (`crop2048_convnext`, `crop2048_convnext_small`, `crop2048_convnext_base --set data.crop_size=768`, `full1024_convnext`, ...; `--set seed=1042` for more seeds) | `runs/<run>/fold*/{best.pt,metrics.csv,oof/}` |
| OOF maps with test-time TTA | `python predict_oof.py --run runs/<run> --out runs/<run>_d4 --tta d4` | `runs/<run>_d4/` |
| blend OOF maps | `python make_ensemble.py --runs runs/A_d4 runs/B_d4 ... --weights ... --out runs/ens` | `runs/ens/` |
| threshold baseline (reference) | `python tune_postprocess.py --run runs/ens` | `postprocess_tuned.yaml` |
| candidate scorer | `python train_scorer.py --run runs/ens` | `scorer.joblib` |
| rotated-crop refiners, 5 folds | `python train_refiner.py --run runs/ens --folds all --encoder resnet34 --epochs 10 --samples-per-epoch 24000 --geom rot` | `refiner/fold*.pt`, `refiner/report.csv` |
| score blend + τ | `python tune_stack.py --run runs/ens` (`--write-oof X.csv`: OOF predictions at the tuned setting) | `stack.yaml` |
| test predictions | `python infer.py --run runs/ens --tta d4` | `submission.csv` |
| score any CSV locally | `python evaluate.py --pred X.csv --gt data/cache/gt_train.csv` | PQ, SQ, RQ, Dice, IoU distribution, 1:n / n:1 |
| error analysis / overlays | `python analyze_oof.py --run R`, `python visualize.py --run R --refiner --zoom 3` | `analysis/`, `viz/` |
| report figure 1, stage timings | `python tools/make_figure_pipeline.py`, `python tools/profile_infer.py --run R` | `runs/figures/` |

`infer.py` picks up whatever a run directory contains: an ensemble definition, a scorer, refiners, the stack settings.
Every stage is validated out-of-fold on the same 5 time-grouped folds, so each tuned number is honest.

Any config key can be overridden from the CLI: `--set train.lr=3e-4 model.encoder=resnet50`.
Configs can inherit (`inherit: baseline.yaml`). Training resumes automatically from `last.pt`.

Smoke test without the real data (about 10 minutes on the laptop GPU):

```bash
python make_synthetic_data.py --out data/synthetic
python prepare_data.py --config configs/smoke.yaml
python train.py --config configs/smoke.yaml --fold all
python tune_postprocess.py --run runs/smoke --eval-size 512
python infer.py --run runs/smoke
```

## Layout

```
configs/        baseline (1024 full disk), crop2048 (512 crops) with ConvNeXt-T/S/B variants, full1024_convnext, smoke
src/
  coco_io.py      MAGFiLO JSON -> observations/annotators/filaments; COCO-RLE helpers
  solar_disk.py   disk detection (GONG geometry prior), limb-darkening flattening
  targets.py      soft multi-annotator targets: mask / boundary / spine heads
  dataset.py      full-disk or positive-biased crop sampling
  transforms.py   albumentations (rotations, photometric, seeing blur, JPEG noise)
  model.py        smp model factory (1-channel input, one output per head, local timm weights)
  losses.py       BCE + soft Dice (+ optional clDice / BiCC) per head
  inference.py    full-image or Hann-blended sliding window, model ensemble, D4 TTA, bf16/fp16 autocast
  postprocess.py  threshold post-processing (resolution-independent pixel units)
  candidates.py   candidate tree, 42 features, exact tree-DP selection, scorer inference
  refiner.py      stage-2 crop refiner (square or rotated crops; outline + score)
  metrics.py      PQ identical to the organizers' notebook (tests/test_metric.py)
  evaluation.py   shared scoring / OOF caching / submission writing
  resources.py    CPU/GPU caps from env vars (SF_CPU_FRAC, SF_GPU_MEM_FRAC, SF_GPU_DUTY)
prepare_data.py train.py predict_oof.py make_ensemble.py tune_postprocess.py train_scorer.py
train_refiner.py tune_stack.py tune_stack_l2.py infer.py evaluate.py analyze_oof.py visualize.py eda.py make_synthetic_data.py
tools/          submit.py (validate/upload), kaggle_cli.py (token-masked Kaggle CLI), detached.ps1 / resume.ps1 / wait_job.ps1
                (long jobs on Windows), auto_stack.py, check_stack.py, profile_infer.py, make_figure_pipeline.py,
                stage_kaggle_dataset.ps1, check_target_geometry.py
notebooks/      reproduce_kaggle.ipynb
docs/           COMPETITION_NOTES.md, EXPERIMENTS.md, HANDOFF.md, report/
```
