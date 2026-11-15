# Agent instructions: SolarFragments

Kaggle **Solar Filament Segmentation Challenge 2026** (`filament-segmentation-2026`), deadline **Nov 15 2026, 06:00 UTC** (midnight at the end of Nov 14, US Central).
The goal is to win: final judging is 70% quantitative (PQ etc.) and 30% qualitative (report, morphology, code quality).
Accuracy comes first; inference speed is secondary but is judged.

Read first: `docs/HANDOFF.md` (current state, next steps), `docs/EXPERIMENTS.md` (every result), `docs/COMPETITION_NOTES.md` (metric, data, rules), `README.md` (pipeline commands).

## Hard rules
- **Never use the public MAGFiLO 1.0 labels** (they overlap the test set and disqualify the entry) or any other extra ground truth. Inference may use only the single supplied test image.
- **Machine**: since 10-27 a dedicated RTX 4090 desktop (32 threads, 24 GB GPU, 64 GB RAM). Since 10-28 the user lifted all limits: caps are 1.0 / 1.0 / 1.0 (`SF_CPU_FRAC`, `SF_GPU_MEM_FRAC`, `SF_GPU_DUTY` user env vars), and a CPU job (ensemble/post-processing tuning) may run alongside one GPU job. Still avoid two GPU jobs at once (a driver hang on 10-16 came from that). Earlier: the laptop (16 threads, 8 GB RTX PRO 2000 Blackwell), the user's daily workstation, at 0.6 / 0.7 / 0.65, one heavy job at a time (it hung twice). Every entry point calls `src/resources.limit_resources()` (below-normal priority, thread/GPU caps from those env vars).
- **Long jobs run detached**, so they survive VS Code / agent restarts:
  `powershell -NoProfile -ExecutionPolicy Bypass -File tools/detached.ps1 -Name <job> -Command "<cmd.exe command line>"`.
  The job writes `runs/jobs/<job>.log` and `runs/jobs/<job>.status`, which holds `RUNNING`, then `DONE <exit code>`.
  After a laptop restart run `powershell -NoProfile -ExecutionPolicy Bypass -File tools/resume.ps1`; jobs resume from their own checkpoints.
- **Kaggle**: always use `python tools/kaggle_cli.py <args>`, which masks the token. Never print `~/.kaggle/access_token` and never run the raw CLI unfiltered (a token once leaked through a traceback). The CLI lives in `.venv-kaggle` (Python 3.14).
  `python tools/submit.py <csv> --upload -m "<msg>"` validates, uploads, waits for the score and logs it to `runs/jobs/submissions.csv`.
  **Standing approval:** upload every finished and validated experiment as a test-set gauge, max 5/day. The user picks the final submission (one selection).
- Python: training venv is `.venv` (Python 3.10, torch 2.11+cu128). Hugging Face downloads are **blocked** on this network: use smp encoders with GitHub/PyTorch URLs (resnet*, efficientnet-b*) or local files in `weights/` (ConvNeXt-T from `dl.fbaipublicfiles.com`).
- Don't commit `data/`, `runs/`, `weights/` or `*.pt` (all gitignored).

## How results are judged here
- Every stage is validated **out-of-fold (OOF)** on 5 folds grouped by 14-day windows (`data/cache/folds.csv`). Downstream stages train only on OOF maps.
- Report **OOF PQ** (organizer-exact, `src/metrics.py`, tested in `tests/`). The public LB is rounded to 2 decimals and noisy (~90 frames, ±0.03), so decide by OOF; OOF → LB has been about −0.045 to −0.06.
- Log every experiment (setup, numbers, conclusion) in `docs/EXPERIMENTS.md`.

## Conventions
- Pixel post-processing parameters are in native 2048-px units (`src/postprocess.to_working_units`).
- Configs inherit (`inherit: baseline.yaml`); CLI overrides go through `--set key=value`.
- Scripts are resumable: `train.py` (last.pt), `infer.py` (`<out>.parts/`), `predict_oof.py` (existing maps), `train_refiner.py` (finished folds).
- Run `python -m pytest tests -q` after changing metrics, selection or post-processing.
