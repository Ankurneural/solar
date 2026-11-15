---
name: resume
description: "Set up SolarFragments on a new machine and resume all experiments from docs/HANDOFF.md"
disable-model-invocation: true
---
This repo (SolarFragments, Kaggle filament-segmentation-2026) was just copied in full from my laptop to this RTX 4090 machine, including `data/`, `runs/` and `weights/`. Pick everything up from here.

1. Read [AGENTS.md](../../AGENTS.md), then [docs/HANDOFF.md](../../docs/HANDOFF.md) (especially "Machine move", "Running right now" and "Next steps"), then [docs/EXPERIMENTS.md](../../docs/EXPERIMENTS.md). Follow every rule in AGENTS.md: one heavy job at a time, long jobs via `tools/detached.ps1`, Kaggle only via `tools/kaggle_cli.py` / `tools/submit.py` (never print the token), never use public MAGFiLO labels or any extra ground truth, decide by OOF PQ, log every result in docs/EXPERIMENTS.md.

2. Set up the environment (the copied venvs point at the laptop's Python and will not work):
   - Report the GPU (`nvidia-smi`), CPU threads, RAM, and the Python versions installed.
   - Rebuild `.venv` with Python 3.10: install torch/torchvision from the cu128 index exactly as in requirements.txt's header, then `pip install -r requirements.txt`. Verify `torch.cuda.is_available()` and that the GPU is the 4090.
   - Rebuild `.venv-kaggle` (Python 3.14 if available, else any 3.11+) with the `kaggle` package. The Kaggle token (`~/.kaggle/access_token`, plain ASCII, no newline) is not in the repo: ask me to put it there myself; never ask me to paste it into chat. Check it with `python tools/kaggle_cli.py competitions submissions -c filament-segmentation-2026` (masked output).
   - Run `python -m pytest tests -q`.

3. Ask me which compute caps to use on this machine (`SF_CPU_FRAC`, `SF_GPU_MEM_FRAC`, `SF_GPU_DUTY` in `src/resources.py`; the laptop used 0.6 / 0.7 / 0.65). Ask whether it is a dedicated machine, then set them as user env vars only after I answer.

4. Resume the interrupted job: `powershell -NoProfile -ExecutionPolicy Bypass -File tools/resume.ps1` (it rewrites old absolute paths in `runs/jobs/*.cmd`). `c2_seed2` should continue from `runs/crop2048_convnext_s2/fold0/last.pt`. Confirm from `runs/jobs/c2_seed2.log` that it resumed (not restarted from epoch 1) and report the epoch speed. If the laptop is still training the same job, tell me so I can stop it there.

5. Continue with "Next steps" in HANDOFF.md, in order, without waiting for my confirmation between steps. Uploading each finished, validated experiment is pre-approved (max 5/day). Use CPU-only gaps for code work, such as the rotated-crop refiner pixel-convention check in step 2. Heavier models (step 3): give me the exact Hugging Face / URL file names to download into `weights/`.

6. Keep docs/HANDOFF.md current (state, running jobs, next steps) whenever something finishes or changes, so the next session can resume from it.
