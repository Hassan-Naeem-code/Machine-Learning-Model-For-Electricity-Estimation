# Per-Asset LightGBM Forecasting

Per-asset gradient-boosted forecasting models for 15-minute-resolution OHLCV data. Trains one LightGBM model per asset, predicts 1–10 step horizons for high / low / close / volume, and produces evaluation summaries.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-gradient_boosting-006aa7)
![License](https://img.shields.io/badge/License-MIT-green)

## What it does

- Trains a **separate LightGBM model per asset** to avoid cross-asset leakage and let hyperparameters adapt to each asset's volatility profile.
- Forecasts **H = 1..10** future steps at **15-minute resolution** for each of `high`, `low`, `close`, and `volume`.
- Supports **incremental retraining** from a checkpoint rather than full re-fits.
- Produces compact CSV summaries and a markdown final report instead of leaving large artifacts in git.

## Why per-asset

A single global model averages behavior across assets with very different volatility, liquidity, and regime characteristics. Per-asset models:

- Keep feature importances interpretable for a single asset's traders.
- Avoid silent degradation when a new asset with unusual behavior is added.
- Let hyperparameters and feature pipelines evolve independently.

Tradeoff: more models to store and retrain. The retrain script keeps this manageable.

## Quick start

```bash
git clone https://github.com/Hassan-Naeem-code/Machine-Learning-Model-For-Electricity-Estimation.git
cd Machine-Learning-Model-For-Electricity-Estimation

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Training — point at your parquet data dir (not committed)
python src/train.py --data-dir data/ --out experiments/

# Retrain a single asset from checkpoint
python src/retrain.py --asset ASSET_ID --data-dir data/
```

## Repository layout

```
.
├── src/                    # training + retraining scripts, feature helpers
├── data/                   # (not committed) OHLCV parquet files per asset
├── experiments/
│   ├── FINAL_REPORT.md                                  # summary of results
│   ├── per_asset_lgbm_results_summary_postproc.csv      # aggregated metrics
│   └── per_asset_lgbm_results_worst20.csv               # bottom-20 assets for diagnosis
├── tests/
├── requirements.txt
└── README.md
```

Large experiment outputs (logs, intermediate parquet, model binaries) are kept out of git. If you need to reproduce a specific run, see `experiments/FINAL_REPORT.md` for configuration and metrics.

## Results (summary)

The full evaluation is in [`experiments/FINAL_REPORT.md`](experiments/FINAL_REPORT.md). Per-horizon and per-asset error metrics live in `experiments/per_asset_lgbm_results_summary_postproc.csv`. The worst-performing 20 assets are broken out in `experiments/per_asset_lgbm_results_worst20.csv` so model failures can be inspected in isolation.

## Reproducing

1. Place asset-level OHLCV parquet files in `data/` (one file per asset, 15-minute rows).
2. Run `python src/train.py --data-dir data/` to produce per-asset models under `experiments/`.
3. Run the evaluation step (`src/evaluate.py` or the notebook under `notebooks/`, if present) to regenerate the summary CSVs.

## License

MIT
