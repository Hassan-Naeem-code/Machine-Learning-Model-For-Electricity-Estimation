#!/usr/bin/env python3
"""
Baselines: zero and persistence, and sMAPE evaluation for multi-horizon forecasts.

This script expects per-asset parquet files in `data/assets_parquet/` produced by the
preprocessing step. It will:
 - create a validation split from the last N days of the training period (configurable)
 - compute predictions for horizons 1..10 using:
   * zero baseline: predict 0 for all horizons and features
   * persistence baseline: predict last observed value for each horizon
 - compute sMAPE per asset and averaged across assets and features
 - save results to `experiments/baseline_results.csv`
"""
from pathlib import Path
import pandas as pd
import numpy as np
import os

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / 'data' / 'assets_parquet'
OUT_DIR = ROOT / 'experiments'
OUT_DIR.mkdir(exist_ok=True)

HORIZON = 10
VAL_DAYS = 28  # last N days of TRAIN considered validation

def smape(y_true, y_pred, eps=1e-8):
    # compute element-wise smape while avoiding any division when denominator is zero
    num = np.abs(y_pred - y_true)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    # result array filled with zeros where den is tiny (treat as zero-error when both are 0)
    res = np.zeros_like(num, dtype=float)
    mask = den > eps
    if np.any(mask):
        # only compute division where safe
        res[mask] = num[mask] / den[mask]
    # return mean across all elements (keep zeros for masked entries)
    return np.nanmean(res) * 100.0

assets = sorted(ASSET_DIR.glob('*.parquet'))
print('Found', len(assets), 'assets')

rows = []
for p in assets[:]:
    id_val = p.stem
    df = pd.read_parquet(p)
    # df index is tz-aware Europe/Berlin
    # determine validation cutoff as last VAL_DAYS before end - but end may include TEST period;
    # we choose validation as last VAL_DAYS within TRAIN years by default: here approximate by
    # using last timestamp minus VAL_DAYS
    # Determine the last timestamp that belongs to TRAIN (2021-2023).
    # Allow overriding via environment variable TRAIN_CUTOFF (ISO string, e.g. 2023-12-31T23:45+01:00
    # or '2023-12-31 23:45'). If not set, default to end of 2023 in Europe/Berlin.
    env_cutoff = os.getenv('TRAIN_CUTOFF')
    if env_cutoff:
        try:
            TRAIN_CUTOFF = pd.to_datetime(env_cutoff).tz_convert('Europe/Berlin') if pd.to_datetime(env_cutoff).tzinfo is not None else pd.to_datetime(env_cutoff).tz_localize('Europe/Berlin')
        except Exception:
            # fallback: parse without timezone then localize
            TRAIN_CUTOFF = pd.to_datetime(env_cutoff).tz_localize('Europe/Berlin')
    else:
        TRAIN_CUTOFF = pd.Timestamp('2023-12-31 23:45', tz='Europe/Berlin')
    # allow overriding validation window via env
    try:
        VAL_DAYS = int(os.getenv('VAL_DAYS', VAL_DAYS))
    except Exception:
        pass
    # last timestamp in this asset that is within TRAIN
    asset_train_last = df.index[df.index <= TRAIN_CUTOFF].max() if (df.index <= TRAIN_CUTOFF).any() else None
    if asset_train_last is None or pd.isna(asset_train_last):
        # no training-period data for this asset -> skip
        continue
    val_start = asset_train_last - pd.Timedelta(days=VAL_DAYS)
    # Build list of evaluation timestamps = those with full horizon available before asset_train_last
    eval_idx = df.index[(df.index >= val_start) & (df.index + pd.Timedelta(minutes=15*HORIZON) <= asset_train_last)]
    if len(eval_idx) == 0:
        continue
    # For each feature
    for feat in ['high','low','close','volume']:
        y_true_list = []
        pred_zero_list = []
        pred_pers_list = []
        for t in eval_idx:
            # create true horizon values
            true_vals = df.loc[t + pd.to_timedelta(np.arange(1, HORIZON+1)*15, unit='m'), feat].values
            # persistence: last observed value at time t
            last_val = df.at[t, feat]
            pred_pers = np.full(HORIZON, last_val)
            pred_zero = np.zeros(HORIZON)
            y_true_list.append(true_vals)
            pred_zero_list.append(pred_zero)
            pred_pers_list.append(pred_pers)
        y_true = np.vstack(y_true_list)
        p_zero = np.vstack(pred_zero_list)
        p_pers = np.vstack(pred_pers_list)
        # compute sMAPE per horizon and average
        smape_zero = smape(y_true, p_zero)
        smape_pers = smape(y_true, p_pers)
        rows.append({'asset': id_val, 'feature': feat, 'baseline': 'zero', 'sMAPE': smape_zero})
        rows.append({'asset': id_val, 'feature': feat, 'baseline': 'persistence', 'sMAPE': smape_pers})

df_res = pd.DataFrame(rows)
outp = OUT_DIR / 'baseline_results.csv'
df_res.to_csv(outp, index=False)
print('Wrote baseline results to', outp)
