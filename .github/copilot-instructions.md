# Copilot instructions

This repository's agent rules live in [`AGENTS.md`](../AGENTS.md); the current state and next plan are in [`docs/HANDOFF.md`](../docs/HANDOFF.md). Read both before doing anything.

Key rules (details in AGENTS.md):
- Run one heavy job at a time, under `src/resources.py` caps. Start long jobs with `tools/detached.ps1`; after a restart, run `tools/resume.ps1`.
- Use Kaggle only via `tools/kaggle_cli.py` / `tools/submit.py` (the token stays masked). Uploading each finished, validated experiment is pre-approved (max 5/day).
- Never use public MAGFiLO labels or any extra ground truth.
- Decide by OOF PQ, and log every experiment in `docs/EXPERIMENTS.md`.
