# MachineLearningModel — submission-ready repository

This repository contains the per-asset LightGBM forecasting baseline and supporting scripts used for the assignment.

Quick status
- Most processing was run locally and produced experiment artifacts under `experiments/`.
- Large logs, parquet files and intermediate artifacts have been moved to a timestamped backup under `experiments/backup_*` to keep the Git working copy small and pushable.

What I included in the repo (cleaned for push)
- Source code: `src/` (training and retrain scripts)
- Small summaries and the final report: `experiments/FINAL_REPORT.md`, `experiments/per_asset_lgbm_results_summary_postproc.csv`, `experiments/per_asset_lgbm_results_worst20.csv`
- A submission package (if present) was moved to backup to avoid committing large binaries — see `experiments/backup_*`.

# MachineLearningModel — Per-asset forecasting (submission-ready)

This repository contains code and small summary artifacts for the per-asset LightGBM forecasting assignment. Large experiment outputs and logs were moved to timestamped backups under `experiments/backup_*` so this repository is small and ready to push.

Contents
- `src/` — training and retrain scripts, helpers
- `data/` — (not committed) asset parquet files; keep these out of git if large
- `notebooks/` — exploratory notebooks (if present)
- `experiments/` — small summaries and `FINAL_REPORT.md`. Large artifacts are in `experiments/backup_*`.

Purpose
- Train LightGBM per-asset models to forecast H=1..10 (15-minute resolution) for features high/low/close/volume.
- Evaluate models with sMAPE and produce per-asset diagnostics.

Status (cleaned for submission)
- Source code: `src/` — included
- Committed summaries: `experiments/per_asset_lgbm_results_summary_postproc.csv`, `experiments/per_asset_lgbm_results_worst20.csv`, `experiments/FINAL_REPORT.md`
- Large outputs (parquet, full logs, large zips) moved to `experiments/backup_*` and are NOT committed to Git.

Requirements
- Python 3.10+ (project used Python 3.14 locally)
- See `requirements.txt` for required packages (pandas, numpy, scikit-learn, lightgbm, etc.)

Quick setup
1. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

2. Smoke test (fast):

```bash
export HORIZON=10 NUM_BOOST_ROUND=10 USE_CLASSIFIER=1 FEATURES=high,low,close,volume MAX_ASSETS=2
.venv/bin/python src/lightgbm_per_asset_baseline.py
```

Full retrain (notes)
- The full retrain (`src/retrain_and_forecast.py`) is computationally expensive and may run for many hours depending on `NUM_BOOST_ROUND` and the number of assets.
- Recommended safe settings to avoid multiprocessing crashes:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 N_WORKERS=1 NUM_BOOST_ROUND=300 HORIZON=10 USE_CLASSIFIER=1
.venv/bin/python src/retrain_and_forecast.py
```

Outputs (where to find them)
- Committed small summaries and the final report:
	- `experiments/per_asset_lgbm_results_summary_postproc.csv`
	- `experiments/per_asset_lgbm_results_worst20.csv`
	- `experiments/FINAL_REPORT.md`
- Full forecasts, large CSVs and logs: `experiments/backup_*/` (do not push these by default)

Known issues and notes
- A previous retrain attempt crashed with a multiprocessing BrokenPipeError; retrains were restarted with N_WORKERS=1 and single-thread BLAS to avoid repeated crashes (safer but slower).
- If you require full forecasts for grading, retrieve them from the backup folder and upload separately (do not add large binaries to the Git repo).

How to prepare the repository for submission (recommended)
1. Stop any long-running jobs on your machine.
2. Keep `experiments/` and large binaries out of Git (`.gitignore` already configured).
3. Upload `experiments/submission_package_final.zip` to your submission portal and push the cleaned repo (source + README + small summaries).

Minimal `.gitignore` entries (applied)
- `.venv/`
- `experiments/`
- `*.parquet`
- `*.log`
- `*.pid`
- `.DS_Store

If you want any changes to the README (more detail, CI, or automated scripts), tell me which sections to expand and I will update it immediately.
