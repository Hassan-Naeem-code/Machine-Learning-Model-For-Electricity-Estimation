#!/usr/bin/env python3
"""
Per-asset LightGBM baseline.

For each asset (up to MAX_ASSETS):
 - build lag features (1..4), hour, dow
 - split TRAIN / VAL inside TRAIN (VAL_DAYS)
 - for each feature and horizon train a small LightGBM regressor on that asset only
 - evaluate sMAPE on VAL

This is slower than pooled but avoids pooling bias.
"""
from pathlib import Path
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / 'data' / 'assets_parquet'
OUT_DIR = ROOT / 'experiments'
OUT_DIR.mkdir(exist_ok=True)

HORIZON = int(os.getenv('HORIZON', 10))
VAL_DAYS = int(os.getenv('VAL_DAYS', 28))
NUM_BOOST_ROUND = int(os.getenv('NUM_BOOST_ROUND', 200))
EARLY_STOPPING_ROUNDS = int(os.getenv('EARLY_STOPPING_ROUNDS', 20))
TRAIN_CUTOFF_ENV = os.getenv('TRAIN_CUTOFF')
if TRAIN_CUTOFF_ENV:
    TRAIN_CUTOFF = pd.to_datetime(TRAIN_CUTOFF_ENV).tz_convert('Europe/Berlin') if pd.to_datetime(TRAIN_CUTOFF_ENV).tzinfo is not None else pd.to_datetime(TRAIN_CUTOFF_ENV).tz_localize('Europe/Berlin')
else:
    TRAIN_CUTOFF = pd.Timestamp('2023-12-31 23:45', tz='Europe/Berlin')

_FEATURES_DEFAULT = ['high', 'low', 'close', 'volume']
# Allow overriding features via env var FEATURES="close,high,low" to skip expensive targets (e.g., volume)
FEATURES_ENV = os.getenv('FEATURES')
if FEATURES_ENV:
    FEATURES = [x.strip() for x in FEATURES_ENV.split(',') if x.strip()]
    print('Using FEATURES from env:', FEATURES)
else:
    FEATURES = _FEATURES_DEFAULT
LAGS = [1,2,3,4]
PRED_THRESH = float(os.getenv('PRED_THRESH', 1e-3))
ACTIVITY_MEAN_THRESH = float(os.getenv('ACTIVITY_MEAN_THRESH', 0.01))
ACTIVITY_STD_THRESH = float(os.getenv('ACTIVITY_STD_THRESH', 0.01))
ACTIVITY_NONZERO_FRAC = float(os.getenv('ACTIVITY_NONZERO_FRAC', 0.05))
USE_CLASSIFIER = int(os.getenv('USE_CLASSIFIER', 1))
CLS_EPS = float(os.getenv('CLS_EPS', 1e-6))


def smape(y_true, y_pred, eps=1e-8):
    num = np.abs(y_pred - y_true)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    res = np.zeros_like(num, dtype=float)
    mask = den > eps
    if np.any(mask):
        res[mask] = num[mask] / den[mask]
    return np.nanmean(res) * 100.0


def run(max_assets=50):
    # Allow restricting to an explicit asset list via env var (one asset id per line)
    asset_list_file = os.getenv('ASSET_LIST_FILE')
    if asset_list_file:
        asset_ids = [x.strip() for x in open(asset_list_file, 'r').read().splitlines() if x.strip()]
        assets = []
        for aid in asset_ids:
            p = ASSET_DIR / f"{aid}.parquet"
            if p.exists():
                assets.append(p)
            else:
                print('Warning: asset file not found for', aid)
        assets = assets[:max_assets]
    else:
        assets = sorted(ASSET_DIR.glob('*.parquet'))[:max_assets]
    rows = []
    for p in assets:
        aid = p.stem
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print('Failed reading', p, e)
            continue
        # only use up to TRAIN_CUTOFF (validation inside TRAIN)
        df_trainpool = df.loc[df.index <= TRAIN_CUTOFF]
        if len(df_trainpool) < max(LAGS) + HORIZON + 10:
            print('Skipping', aid, 'not enough data')
            continue
        train_cut = TRAIN_CUTOFF - pd.Timedelta(days=VAL_DAYS)
        train_df = df_trainpool.loc[df_trainpool.index <= train_cut].copy()
        val_df = df_trainpool.loc[(df_trainpool.index > train_cut) & (df_trainpool.index <= TRAIN_CUTOFF)].copy()
        if len(train_df) < 50 or len(val_df) < 10:
            print('Skipping', aid, 'not enough train/val samples', len(train_df), len(val_df))
            continue
        # compute simple activity stats per feature on train_df
        activity = {}
        for f in FEATURES:
            vals = train_df[f].dropna().values
            if len(vals) == 0:
                activity[f] = {'mean_abs':0.0, 'std':0.0, 'nonzero_frac':0.0}
            else:
                activity[f] = {
                    'mean_abs': float(np.mean(np.abs(vals))),
                    'std': float(np.std(vals)),
                    'nonzero_frac': float(np.mean(np.abs(vals) > CLS_EPS))
                }
        # build lags and features
        for lag in LAGS:
            for f in FEATURES:
                train_df[f'f_lag{lag}'] = train_df[f].shift(lag)
                val_df[f'f_lag{lag}'] = val_df[f].shift(lag)
        train_df['hour'] = train_df.index.hour
        train_df['dow'] = train_df.index.dayofweek
        val_df['hour'] = val_df.index.hour
        val_df['dow'] = val_df.index.dayofweek

        feature_cols = [f'f_lag{lag}' for lag in LAGS] + ['hour','dow']

        for feat in FEATURES:
            is_active = (activity[feat]['mean_abs'] >= ACTIVITY_MEAN_THRESH) or (activity[feat]['std'] >= ACTIVITY_STD_THRESH) or (activity[feat]['nonzero_frac'] >= ACTIVITY_NONZERO_FRAC)
            if not is_active:
                print(f'Asset {aid} feat {feat} flagged inactive (mean_abs={activity[feat]["mean_abs"]:.6g}, std={activity[feat]["std"]:.6g}, nonzero_frac={activity[feat]["nonzero_frac"]:.3f}) -> predict zeros')

            for h in range(1, HORIZON+1):
                y_col = f'y_h{h}'
                tr = train_df.copy()
                va = val_df.copy()
                tr[y_col] = tr[feat].shift(-h)
                va[y_col] = va[feat].shift(-h)
                tr = tr.dropna(subset=[y_col])
                va = va.dropna(subset=[y_col])
                if len(tr) < 50 or len(va) < 10:
                    continue
                X_tr = tr[feature_cols]
                y_tr = tr[y_col]
                X_va = va[feature_cols]
                y_va = va[y_col]

                # If feature flagged inactive: predict zeros (baseline) and skip training
                if not is_active:
                    y_pred = np.zeros_like(y_va.values)
                    score = smape(y_va.values, y_pred)
                    rows.append({'asset':aid, 'feature':feat, 'horizon':h, 'sMAPE':score, 'n_train':len(tr), 'n_val':len(va)})
                    print(f'Asset {aid} feat {feat} h{h} sMAPE {score:.3f} (inactive)')
                    continue

                # Active: train regressor. Optionally train a small classifier to gate zeros (for sparse assets)
                lgb_tr = lgb.Dataset(X_tr, y_tr)
                lgb_va = lgb.Dataset(X_va, y_va, reference=lgb_tr)
                params = {'objective':'regression', 'metric':'l2', 'verbosity':-1, 'num_leaves':31, 'learning_rate':0.05, 'seed':42}
                bst = lgb.train(params, lgb_tr, num_boost_round=NUM_BOOST_ROUND, valid_sets=[lgb_va], callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS), lgb.log_evaluation(period=0)])
                y_pred = bst.predict(X_va, num_iteration=bst.best_iteration)

                if USE_CLASSIFIER:
                    # train a small logistic regressor to predict whether target is non-zero
                    y_bin = (np.abs(y_tr.values) > CLS_EPS).astype(int)
                    # Only train classifier if there is some class variability
                    if y_bin.sum() > 5 and (y_bin.sum() < len(y_bin) - 5):
                        try:
                            clf = LogisticRegression(max_iter=200, class_weight='balanced')
                            clf.fit(X_tr.fillna(0.0).values, y_bin)
                            proba = clf.predict_proba(X_va.fillna(0.0).values)[:, 1]
                            mask_nonzero = proba >= 0.5
                            # apply classifier gating: where predicted zero, set pred to 0
                            y_pred = np.where(mask_nonzero, y_pred, 0.0)
                        except Exception as e:
                            # classifier failed — fall back to regressor predictions
                            print('Classifier failed for', aid, feat, h, e)

                # apply absolute-threshold post-processing: set tiny preds to zero to reduce sMAPE blow-ups
                if PRED_THRESH is not None and PRED_THRESH > 0:
                    y_pred = np.where(np.abs(y_pred) < PRED_THRESH, 0.0, y_pred)

                score = smape(y_va.values, y_pred)
                rows.append({'asset':aid, 'feature':feat, 'horizon':h, 'sMAPE':score, 'n_train':len(tr), 'n_val':len(va)})
                print(f'Asset {aid} feat {feat} h{h} sMAPE {score:.3f}')
    out = pd.DataFrame(rows)
    outp = OUT_DIR / 'per_asset_lgbm_results.csv'
    out.to_csv(outp, index=False)
    print('Wrote per-asset results to', outp)


if __name__ == '__main__':
    max_assets = int(os.getenv('MAX_ASSETS', 50))
    run(max_assets=max_assets)
