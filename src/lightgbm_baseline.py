#!/usr/bin/env python3
"""
Pooled LightGBM baseline.

Strategy:
 - Pool data across assets (include asset id as categorical)
 - Build lag features (lags 1..4) for each feature separately
 - Use direct forecasting: for each horizon h (1..HORIZON) train a separate model to predict value at t+h
 - Train on TRAIN (<= TRAIN_CUTOFF - VAL_DAYS), validate on last VAL_DAYS within TRAIN

This is intentionally modest (small model/early stopping) to be runnable in the project venv.
"""
from pathlib import Path
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / 'data' / 'assets_parquet'
OUT_DIR = ROOT / 'experiments'
OUT_DIR.mkdir(exist_ok=True)

HORIZON = int(os.getenv('HORIZON', 10))
VAL_DAYS = int(os.getenv('VAL_DAYS', 28))
TRAIN_CUTOFF_ENV = os.getenv('TRAIN_CUTOFF')
if TRAIN_CUTOFF_ENV:
    TRAIN_CUTOFF = pd.to_datetime(TRAIN_CUTOFF_ENV).tz_convert('Europe/Berlin') if pd.to_datetime(TRAIN_CUTOFF_ENV).tzinfo is not None else pd.to_datetime(TRAIN_CUTOFF_ENV).tz_localize('Europe/Berlin')
else:
    TRAIN_CUTOFF = pd.Timestamp('2023-12-31 23:45', tz='Europe/Berlin')

# features to model
FEATURES = ['high', 'low', 'close', 'volume']
LAGS = [1,2,3,4]

def smape(y_true, y_pred, eps=1e-8):
    num = np.abs(y_pred - y_true)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    res = np.zeros_like(num, dtype=float)
    mask = den > eps
    if np.any(mask):
        res[mask] = num[mask] / den[mask]
    return np.nanmean(res) * 100.0

def build_dataset(max_assets=100):
    assets = sorted(ASSET_DIR.glob('*.parquet'))[:max_assets]
    rows = []
    asset_ids = []
    for p in assets:
        aid = p.stem
        df = pd.read_parquet(p)
        # only keep rows up to TRAIN_CUTOFF (we'll make validation inside train)
        # take an explicit copy to avoid SettingWithCopyWarning when assigning new cols
        df_train = df.loc[df.index <= TRAIN_CUTOFF].copy()
        if len(df_train) < max(LAGS)+HORIZON+10:
            continue
        # compute lags
        for lag in LAGS:
            for f in FEATURES:
                df_train[f'f_lag{lag}'] = df_train[f].shift(lag)
        df_train['hour'] = df_train.index.hour
        df_train['dow'] = df_train.index.dayofweek
        df_train['asset'] = aid
        rows.append(df_train.dropna())
        asset_ids.append(aid)
    if len(rows)==0:
        raise RuntimeError('No data assembled for LightGBM baseline')
    big = pd.concat(rows)
    # label encode asset
    le = LabelEncoder()
    big['asset_code'] = le.fit_transform(big['asset'])
    return big, le

def train_and_eval(max_assets=100):
    big, le = build_dataset(max_assets=max_assets)
    results = []
    # define train/val split by timestamp within TRAIN_CUTOFF: val is last VAL_DAYS
    train_cut = TRAIN_CUTOFF - pd.Timedelta(days=VAL_DAYS)
    # use copies to avoid chained-assignment warnings and ensure safe column writes
    train_df = big.loc[big.index <= train_cut].copy()
    val_df = big.loc[(big.index > train_cut) & (big.index <= TRAIN_CUTOFF)].copy()
    print('Pooled samples: train', len(train_df), 'val', len(val_df))
    if len(val_df)==0 or len(train_df)==0:
        raise RuntimeError('Not enough train/val samples')

    feature_cols = [f'f_lag{lag}' for lag in LAGS] + ['hour','dow','asset_code']

    for feat in FEATURES:
        # compute per-asset statistics (mean/std) from the TRAIN portion for this feature
        asset_stats = train_df.groupby('asset')[feat].agg(['mean', 'std']).reset_index().set_index('asset')
        mean_map = asset_stats['mean'].to_dict()
        std_map = asset_stats['std'].to_dict()
        # clip tiny stds to avoid divide-by-zero
        for k, v in list(std_map.items()):
            if not np.isfinite(v) or v < 1e-6:
                std_map[k] = 1.0
        # prepare X/y for each horizon
        for h in range(1, HORIZON+1):
            y_col = f'y_h{h}'
            # build target on copies for this horizon to avoid contaminating other horizons
            tr = train_df.copy()
            va = val_df.copy()
            tr[y_col] = tr[feat].shift(-h)
            va[y_col] = va[feat].shift(-h)
            tr = tr.dropna(subset=[y_col])
            va = va.dropna(subset=[y_col])
            # keep original (denormalized) y for final scoring
            y_va_orig = va[y_col].copy()
            # apply per-asset normalization to lag features and the target
            # map means/stds to rows
            tr_asset_mean = tr['asset'].map(mean_map)
            tr_asset_std = tr['asset'].map(std_map)
            va_asset_mean = va['asset'].map(mean_map)
            va_asset_std = va['asset'].map(std_map)
            # normalize lag features
            for lag in LAGS:
                col = f'f_lag{lag}'
                tr[col] = (tr[col] - tr_asset_mean) / tr_asset_std
                va[col] = (va[col] - va_asset_mean) / va_asset_std
            # normalize target
            tr[y_col] = (tr[y_col] - tr_asset_mean) / tr_asset_std
            va[y_col] = (va[y_col] - va_asset_mean) / va_asset_std
            if len(tr) < 100 or len(va) < 20:
                continue
            X_tr = tr[feature_cols]
            y_tr = tr[y_col]
            X_va = va[feature_cols]
            y_va = va[y_col]

            # sample-weight by absolute (denormalized) target magnitude so model focuses on larger values
            # use small epsilon to avoid zero weights
            w_tr = np.abs(tr[y_col].values) + 1e-6
            w_va = np.abs(va[y_col].values) + 1e-6
            lgb_train = lgb.Dataset(X_tr, y_tr, weight=w_tr)
            lgb_val = lgb.Dataset(X_va, y_va, weight=w_va, reference=lgb_train)
            params = {
                'objective':'regression', 'metric':'l2', 'verbosity':-1,
                'num_leaves':31, 'learning_rate':0.05, 'seed':42
            }
            # two-stage: binary classifier for zero vs non-zero, then regressor for positive values
            # Recover original (denormalized) target for binary label creation
            tr_orig_y = tr[y_col].values * tr_asset_std.values + tr_asset_mean.values
            va_orig_y = va[y_col].values * va_asset_std.values + va_asset_mean.values
            eps_nonzero = 1e-8
            y_tr_bin = (np.abs(tr_orig_y) > eps_nonzero).astype(int)
            y_va_bin = (np.abs(va_orig_y) > eps_nonzero).astype(int)

            # train classifier on same features (normalized)
            cls_params = {'objective':'binary', 'metric':'binary_logloss', 'verbosity':-1, 'learning_rate':0.05, 'num_leaves':31, 'seed':42}
            lgb_cls_train = lgb.Dataset(X_tr, y_tr_bin)
            lgb_cls_val = lgb.Dataset(X_va, y_va_bin, reference=lgb_cls_train)
            bst_cls = lgb.train(cls_params, lgb_cls_train, num_boost_round=100, valid_sets=[lgb_cls_val], callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=0)])

            # train regressor (on normalized target)
            w_tr = np.abs(tr[y_col].values) + 1e-6
            w_va = np.abs(va[y_col].values) + 1e-6
            lgb_train = lgb.Dataset(X_tr, y_tr, weight=w_tr)
            lgb_val = lgb.Dataset(X_va, y_va, weight=w_va, reference=lgb_train)

            bst = lgb.train(
                params,
                lgb_train,
                num_boost_round=200,
                valid_sets=[lgb_val],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=20),
                    lgb.log_evaluation(period=0),
                ],
            )
            y_pred = bst.predict(X_va, num_iteration=bst.best_iteration)
            # invert normalization for predictions: y_pred is in standardized units
            va_std_rows = va['asset'].map(std_map).values
            va_mean_rows = va['asset'].map(mean_map).values
            y_pred_denorm = y_pred * va_std_rows + va_mean_rows
            # classifier predictions -> mask
            y_pred_cls_prob = bst_cls.predict(X_va, num_iteration=bst_cls.best_iteration)
            mask_pos = y_pred_cls_prob >= 0.5
            y_final = np.where(mask_pos, y_pred_denorm, 0.0)
            # compute sMAPE against original (denormalized) validation target
            score = smape(y_va_orig.values, y_final)
            # Debug: for the first feature & horizon, print a small sample of X/y/pred
            if feat == FEATURES[0] and h == 1:
                try:
                    sample_n = min(5, len(X_va))
                    print('\n--- DEBUG: sample validation inputs (first rows) ---')
                    # show small readable JSON for the first rows of X_va
                    sample_X = X_va.iloc[:sample_n].astype(float).to_dict(orient='records')
                    print('X_va sample:', json.dumps(sample_X, default=float))
                    print('y_va sample:', y_va.iloc[:sample_n].to_list())
                    print('y_pred sample:', y_pred[:sample_n].tolist())
                    print('X_va mean/std per col:')
                    print((X_va.mean()).to_dict())
                    print((X_va.std()).to_dict())
                    print('y_va mean/std:', float(y_va.mean()), float(y_va.std()))
                    print('y_pred mean/std:', float(np.mean(y_pred)), float(np.std(y_pred)))
                    print('--- END DEBUG ---\n')
                except Exception as e:
                    print('DEBUG print failed:', e)
            score = smape(y_va_orig.values, y_pred_denorm)
            results.append({'feature':feat, 'horizon':h, 'sMAPE': score})
            print(f'Feat {feat} h{h} sMAPE {score:.3f}')

    out = pd.DataFrame(results)
    outp = OUT_DIR / 'lgbm_results.csv'
    out.to_csv(outp, index=False)
    print('Wrote LightGBM results to', outp)

if __name__ == '__main__':
    # limit assets to keep runtime reasonable; override via env MAX_ASSETS
    max_assets = int(os.getenv('MAX_ASSETS', 50))
    train_and_eval(max_assets=max_assets)
