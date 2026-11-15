# Handoff: state as of 2026-11-11 (training closed; repo cleaned for the move back to the laptop)

## Where we are
| submission | pipeline | OOF PQ | public LB |
|---|---|---|---|
| v1 | B0 threshold | 0.363 | 0.31 |
| v2 | B0 + GBM scorer | 0.378 | 0.32 |
| v3 | ENS1 stack (C1+B0, scorer, refiners) | 0.4244 | 0.38 |
| v4 | W712 stack (0.7 C2 / 0.1 C1 / 0.2 B0) | 0.4501 | 0.39 |
| v5 | W71_B1 stack (0.7 C2 / 0.1 C1 / 0.2 B1) | 0.4542 | 0.39 |
| v6 | W71_B1 stack, ResNet34 refiners | 0.4569 | 0.40 |
| v7 | W71_B1 stack, rotated-crop R34 refiners (`runs/w71_b1_rot`) | 0.4623 | 0.40 |
| v8 | 0.4 C2 / 0.4 C2 seed 2 / 0.2 B1 stack, borrowed rot refiners (`runs/w_s2_44`) | 0.4647 | 0.40 |
| v9 | w9_all (3 C2 seeds, C3S, 2 B1 seeds) stack, borrowed rot refiners (`runs/w9_all`) | 0.4654 | 0.40 |
| **v11** | **w11_all_768b (+ C768, C768B) stack, borrowed rot refiners** (`runs/w11_all_768b`) | **0.4667** | **0.40** |

Honest public scores are ~0.35–0.50; anything above ~0.45 on the LB is leak-driven (see COMPETITION_NOTES).
The current best pipeline is `runs/w71_b1_r34/` (same as `runs/w71_b1/` except the refiners):
- `ensemble.yaml`: members and weights
- `scorer.joblib`: GBM, merge-0 candidate tree
- `refiner/fold*.pt`: 5 ResNet34 refiners (10 ep x 24k)
- `stack.yaml`: GBM/refiner score blend 0.3, keep threshold τ 0.30

Reproduce the submission:
`python infer.py --run runs/w71_b1_r34 --tta d4 --out <csv>` (~45 s/frame, ~2.3 h for the 180 test frames).

## Pipeline (details: README.md)
```
prepare_data → stage-1 train.py ×5 folds per model:
   C2 crop2048_convnext (ConvNeXt-T, 512 crops @2048)   runs/crop2048_convnext
   C1 crop2048          (ResNet34, 512 crops @2048)     runs/crop2048
   B1 full1024_convnext (ConvNeXt-T, full disk @1024)   runs/full1024_convnext
→ predict_oof.py --tta d4   (runs/*_d4)
→ make_ensemble.py           (runs/w71_b1)
→ train_scorer.py            (candidate tree + GBM)
→ train_refiner.py --folds all   (per-candidate crop U-Net: outline + score)
→ tune_stack.py              (score blend + tau)
→ infer.py → tools/submit.py --upload
```

## Machine (since 10-27): RTX 4090 desktop, dedicated
i9-14900KF (32 threads), 64 GB RAM, RTX 4090 24 GB, driver 560.94. Same repo path as on the laptop.
- `.venv` was rebuilt with Python 3.10.11 (installed via `pymanager install 3.10`), torch 2.11.0+cu128 and requirements.txt; tests pass.
- `.venv-kaggle` works as copied (its base, Python 3.14 at `%LOCALAPPDATA%\Python\pythoncore-3.14-64`, exists here too).
- Kaggle token is in place (`~/.kaggle/access_token`, verified 10-27 with `tools/kaggle_cli.py competitions submissions`).
- **No limits (user, 10-28):** `SF_CPU_FRAC=1.0`, `SF_GPU_MEM_FRAC=1.0`, `SF_GPU_DUTY=1.0`; a CPU tuning job may run next to one GPU job (never two GPU jobs).
  `src/resources.py` reads them from `HKCU\Environment` if they are not in the process env, and every job log starts with a `[resources]` line.
- Hugging Face is still blocked (403) on this network; `dl.fbaipublicfiles.com` and GitHub work.
  Downloaded to `weights/`: `convnext_small_22k_1k_224.pth`, `convnext_base_22k_1k_224.pth` (both verified to load exactly).
- Speed: C2 stage-1 trains at ~1.0 min/epoch (40 epochs ≈ 40 min/fold); D4 OOF for one C2 run 8 min; R34 refiner ~35 min/fold
  (CPU-bound with 4 loader workers; `train_refiner.py --workers N` now sets them, default 8).
- Queued jobs wait on their predecessor with `tools/wait_job.ps1 -Name <job>` (each has its own log/status; `resume.ps1` handles them).

## Results of the 10-27 overnight gates (fold 0; details in EXPERIMENTS.md)
| gate | fold-0 PQ | reference | delta | decision |
|---|---|---|---|---|
| rotated-crop R34 refiner (`refiner_rot`) | **0.4532** (SQ 0.684) | square R34 0.4482 | **+0.0051** | adopted, folds 1-4 training |
| square refiner, −0.5 px target fix (`refiner_sqfix`) | 0.4477 | 0.4482 | −0.0004 | no gain (fix kept as code default) |
| ConvNeXt-S stage-1 alone (`f0_C3S`, D4, tuned) | 0.4290 | C2 `f0_C2` 0.4314 | −0.0024 | no gain alone; untested as an extra member |
| C2 with stage-1 target offset −0.5 (`f0_C2T`) | 0.4307 | 0.4314 | −0.0007 | noise; not adopted |

## Status 11-11: training closed, repo cleaned, moving back to the laptop
- **No jobs running, no more training.** Deadline: see the 11-12 notes below (midnight at the end of Nov 14, Central). Both candidates are uploaded; one can be selected:
  A = v11 `runs/w11_all_768b/submission_w11_stack.csv` (OOF 0.4667, LB 0.40, 18.7 s/img), B = `runs/w_s2_44h/submission_fast_hflip.csv`
  (OOF 0.4623, LB 0.40, 3.3 s/img).
- **Cleanup 11-11** (user-approved; runs/ 130 GB → 8.7 GB). Kept: the 8 member runs' `fold*/best.pt` + configs + metrics, their D4 OOF maps
  (`*_d4`) and B's hflip maps (`*_hf`), both final run dirs complete, `runs/w71_b1/fold*/oof` + `runs/w71_b1_r34/refiner/candidates.pkl`
  + `refiner_rot/` (inputs/outputs of the deployed refiners), `runs/{report,figures,ablation,speed,kaggle_kernel}`, every submitted CSV,
  `runs/jobs/submissions.csv`, and small text files (metrics.csv, yaml, report.csv) of every experiment. Deleted: all `last.pt`, the test cache,
  best.pt/OOF maps of non-final models, unused refiners, caches, logs, scratch. data/, weights/ and the competition zip are kept.
  Verified after cleanup: tests pass; both finals re-run on 3 test frames match the submitted CSVs (IoU ≥ 0.999).
- **On the laptop:** do not copy `.venv`/`.venv-kaggle` (machine-specific); rebuild per README. Put the Kaggle token back in `~/.kaggle/access_token`.
  The laptop GPU is Blackwell: same cu128 wheels. Restore the laptop caps (0.6/0.7/0.65) in its env vars if it is the daily machine again.
- **Kaggle reproduction:** dataset `ankurparmar/solarfragments-weights` (PRIVATE, v3) + notebook `ankurparmar/solarfragments-reproduce` (PRIVATE).
  Run v5 (11-09): **B completed** on the T4 (1,111 filaments vs 1,109 submitted; 99.5 % of filaments IoU ≥ 0.95 with the submitted file; the
  rest are fp16-vs-bf16 near-threshold flips); **A hit Kaggle's 12 h limit at 73/180 frames** (~8 min/frame on a T4: 40 models × D4).
  Open decision for the user: run A in a separate session/notebook split by frame ranges, run A with hflip (OOF unchanged on D4 maps
  ≈ v11, ~3x faster), or document A as GPU-only (RTX 4090: 56 min) and reproduce only B on Kaggle. Then ask the user before making
  dataset + notebook **public** (required for judges). Rebuild the dataset folder with `tools/stage_kaggle_dataset.ps1` if code changes.
- **Remaining deliverables:** report (`docs/report/draft_sections.tex` → user's Overleaf; Figure 1 `runs/figures/fig1_pipeline.pdf`,
  Figure 2 crops in `runs/figures/v11_refined/`; a final consistency pass of every number against EXPERIMENTS.md), Kaggle notebook decision above,
  public GitHub repo (user handles git; LICENSE MIT present; .gitignore excludes data/runs/weights/zip).
## Laptop state (11-11 evening; the repo folder was renamed)
- `.venv` (Python 3.10.4, torch 2.11.0+cu128) and `.venv-kaggle` (Python 3.14, kaggle 2.2.5) rebuilt; 19 tests pass; Kaggle CLI verified.
  `SF_*` env vars are unset, so the laptop defaults 0.6 / 0.7 / 0.65 apply.
- **TLS interception on this network.** The venv's stock pip 22 fails with `CERTIFICATE_VERIFY_FAILED`; pip ≥ 24.2 uses the
  Windows cert store. Bootstrap: `.venv-kaggle\Scripts\python -m pip download pip --no-deps -d <dir> --python-version 3.10 --only-binary=:all:`,
  then `.venv\Scripts\python -m pip install --no-index <dir>\pip-*.whl`. `.venv-kaggle` also needs `pip install six pip-system-certs`
  (kaggle 2.x imports `six`; `pip-system-certs` makes `requests` trust the Windows store). Never disable verification.
- B re-run on 3 test frames matches the submitted CSV (IoU ≥ 0.9985). **A does not fit the 8 GB GPU** (40 checkpoints loaded up front);
  run A only on the 4090 or a ≥ 16 GB GPU. Both submitted CSVs re-validate (`tools/submit.py`).
- A vs B paired on OOF: +0.0045, 95% interval +0.0016 to +0.0073 (EXPERIMENTS.md). Report draft updated: \todo{} left only for the repo URL.
- Test images are `.jpeg` (180 files).
- **11-12: only ONE final selection is possible** (user saw it in the Kaggle UI; the "2 final selections" in COMPETITION_NOTES came
  from competitors' mirrors, not from Kaggle). Pick **A** (`submission_w11_stack.csv`, ref 57901450); B stays in the report as the
  efficient variant. Select it explicitly: every candidate ties at 0.40, so Kaggle's automatic pick is arbitrary.
- **11-12: the Kaggle API gives the deadline as `2026-11-15T06:00:00Z` = Nov 15 00:00 CST, i.e. midnight at the END of Nov 14**
  Central, not "Nov 15 23:59 CDT" as written elsewhere in these docs. Treat the evening of Nov 14 as the deadline unless the
  Overview page says otherwise for the report/repo.

## Publication plan (user decision 11-12): go public on the evening of Nov 14, not earlier
**DO NOT PUBLISH ANYTHING (dataset, notebook, GitHub repo) until the user returns on Saturday Nov 14 evening and says so.**
Publishing early would hand every team our checkpoints and a 0.40 notebook. Until then both stay PRIVATE.
Deadline (Kaggle API): Nov 15 00:00 CST = midnight at the end of Saturday Nov 14. The report and the public GitHub repo link are
submitted through a separate Google form from the organizers (user has it); check that form's own due date.
- Notebook v6 (pushed 11-12): B end to end by default (`RUN_A = False`), A documented as local-GPU (≥ 16 GB; 56 min on the 4090).
  Verify its output against `runs/w_s2_44h/submission_fast_hflip.csv` once the run completes
  (`python tools/kaggle_cli.py kernels output ankurparmar/solarfragments-reproduce -p <dir>`).
- **Nov 14 evening (Central), before 23:59:**
  1. Dataset: `python tools/kaggle_cli.py datasets metadata ankurparmar/solarfragments-weights -p <dir>`, set `"isPrivate": false` in
     `<dir>/dataset-metadata.json`, then the same command with `--update` (untested; fallback: dataset page → Settings → Sharing → Public).
  2. Notebook: Kaggle UI → Share → Public (a CLI push with `"is_private": false` also works but re-runs the notebook, ~1-2 h of GPU).
  3. Check both pages in a signed-out browser window, and that the notebook still lists the dataset as its input.
- **11-14 20:05 CST: dataset `ankurparmar/solarfragments-weights` is PUBLIC** (user said "publish"; `datasets metadata --update` with
  `"isPrivate": false` worked; signed-out HTTP 200). Notebook v6 completed on the T4 (B: 1,111 filaments, 99.5% IoU ≥ 0.95 vs the
  submitted file, 41.6 s/image). **11-14 20:11 CST: notebook `ankurparmar/solarfragments-reproduce` is PUBLIC too** (user toggled it
  in the UI; signed-out HTTP 200, live metadata `is_private: false`, input dataset attached). The local
  `runs/kaggle_kernel/kernel-metadata.json` now says `"is_private": false`, so a later push does not make it private again.
- **11-14 20:36 CST: public GitHub repo pushed**: https://github.com/Ankurneural/solar (branch `main`; author is the GitHub no-reply
  address of the personal account; the remote URL names `Ankurneural` so the work account's stored credential is never used).
- Also due by then: final selection of A on Kaggle (ref 57901450), report on Overleaf, public GitHub repo.

## Final decision (11-12: one selection only, so A is the final submission and B the efficient variant)
- A: v11 (`runs/w11_all_768b`, D4), best OOF 0.4667, 18.7 s/frame.
- B: fast pipeline `runs/w_s2_44h` with `--tta hflip` (3 members tuned on hflip OOF maps), OOF 0.4623, 3.3 s/frame, LB 0.40.
## Next steps
See "Remaining deliverables" above. Model work is closed (stage 1 plateaued at ~0.453 threshold OOF, the stack at ~0.466).

## Submission strategy (5/day; the quota resets at 00:00 UTC = 19:00 CDT, 18:00 CST from Nov 1)
Unused slots are lost, so plan them before 19:00 every day. Log every upload in `runs/jobs/submissions.csv` (submit.py does) and EXPERIMENTS.md.
What the public LB can and cannot tell us:
- ~90 frames, rounded to 2 decimals. A paired comparison of two submissions on the same frames resolves only effects ≥ ~0.01.
  Tuning knobs (tau 0.26–0.30 is flat to 0.0002 on OOF) are invisible, so **never tune on the LB; OOF stays the decision metric**.
- It is the only check of things OOF cannot measure: the 5-fold test ensemble vs single-model OOF maps (calibration shift),
  the test distribution, and bugs in the inference path.
- Leaderboard ranks are meaningless (top 0.56–0.70 is the MAGFiLO leak); honest public pipelines sit at 0.36–0.40.
Slot policy, in priority order:
1. **Every OOF improvement** of the full pipeline (≥ 0.002 over the previous best) → upload as vN. Expect ~1 per day.
2. **Test-only decisions with effects that may reach ≥ 0.01**: tau probes far from the OOF optimum (e.g. 0.22 / 0.38, to detect a
   calibration shift of the 5-fold ensemble), hflip vs D4 TTA and 3 vs 5 folds (inference speed is judged), refiner on/off.
3. **Test-set ablations for the report** (30% of the grade): threshold-only vs +scorer vs +refiner, single members vs the ensemble.
4. Fill the remaining slots before 19:00 with item 3.
`infer.py` caches every member's stage-1 test maps (`runs/test_cache/<member>__<tta>__<size>__<ckpt hash>/`, float16, identical
output with or without the cache). Only the first submission of a new member pays for stage 1; selection, refiner and ensemble-weight
variants then cost ~5–10 min. Final picks (user): the best-OOF pipeline, plus a hedge that differs in a test-only choice where the LB disagrees.

## Things already tried that did NOT help (don't repeat)
- BiCC object-level loss on C2 (fold 0): 0.4317 vs 0.4314 after tuning; only shifts calibration.
- Stage-1 target pixel offset −0.5 (fold 0): 0.4307 vs 0.4314. Square-refiner target offset fix: −0.0004.
- ConvNeXt-S alone (fold 0): 0.4290 vs 0.4314 (still being tested as an extra ensemble member).
- Heliographic |lon|/|lat| scorer features: +0.0004.
- Scorer member-disagreement features: 0.4448 vs 0.4457.
- Scorer as P(match) × E[IoU]: 0.4462 vs 0.4470.
- Candidate merge 10/20 px: merge 0 is best. 18 threshold levels instead of 10: no gain.
- Both B0 and B1 together in the blend: worse than B1 alone. B0 is replaced by B1.
- EfficientNet-B4 stage-1: the run hung at the first batch while workers started (parked, not evaluated).
- clDice / Lovász losses: avoided (public reports say they hurt calibration).
- The `boundary_cut` and spine-seeded grouping post-processing options were never selected by tuning.

## Gotchas learned the hard way
- **Never hide the GPU with `CUDA_VISIBLE_DEVICES=-1`** on this machine: Python then crashes natively (0xC0000005) in
  `nvdxgdmal64.dll_unloaded` (NVIDIA driver 560.94) at random points (OpenCV, pickling, conv). This, not our code, caused the
  "CPU-only infer crashes" of 10-29. Normal GPU runs are unaffected. For CPU work, leave the GPU visible.
- The console window of a detached job can be closed by mistake ("forrtl: window-CLOSE event"); relaunch with the same
  `detached.ps1` command, since training resumes from `last.pt`.
- Machine note: i9-14900KF, MSI MEG Z790 ACE BIOS 1.A1 (2023-09), CPU microcode 0x11D, which predates Intel's 0x129+ fixes for
  13th/14th-gen instability. No WHEA errors so far; a BIOS update is the user's call.
- Never train refiners or run tune_stack in a deployed run dir's `refiner/`: a worse result overwrites the submitted pipeline.
  Use `--tag` and a copied run dir (v10 overwrote v9's files on 11-01; restored).
- `infer.py` on CPU (`CUDA_VISIBLE_DEVICES=-1`, threshold-only from the test cache) crashes natively (0xC0000005) every ~70 frames;
  the per-frame `.parts` checkpoints make a rerun resume. Never seen on GPU. Investigate before the CPU reproduction notebook.
- The `cmd` status line must redirect first: `>"file" echo DONE %errorlevel%`. Written as `echo DONE 0> file`, it writes nothing.
- `cudnn.benchmark` re-autotunes on every new input shape, so keep batch shapes fixed (refiner batches are padded to 64).
- The GPU throttle must bracket only GPU work (`begin()` / `step()`).
- Windows PowerShell's `>` writes UTF-16. The Kaggle token file must be plain ASCII with no newline.
- A full OS restart kills even detached jobs. Use `tools/resume.ps1`.
- Refiner training saves `<fold>.pt.last` every epoch and reuses finished `<fold>.pt` (resume after restarts).
- Never edit a running job's `.cmd`: cmd.exe reads batch files incrementally.
- Scripts run from `runs/` need the repo root on `PYTHONPATH`.
