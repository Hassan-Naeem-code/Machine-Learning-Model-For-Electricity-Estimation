#!/usr/bin/env python3
"""
Retrain per-asset models on full TRAIN (up to TRAIN_CUTOFF) and forecast for TEST (after TRAIN_CUTOFF).
Writes:
 - experiments/test_forecasts.parquet: columns [asset, feature, origin, horizon, target_time, pred, true]
 - experiments/per_asset_test_results.csv: aggregated sMAPE per asset/feature/horizon
"""
from pathlib import Path
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / 'data' / 'assets_parquet'
OUT_DIR = ROOT / 'experiments'
OUT_DIR.mkdir(exist_ok=True)

HORIZON = int(os.getenv('HORIZON', 10))
NUM_BOOST_ROUND = int(os.getenv('NUM_BOOST_ROUND', 100))
TRAIN_CUTOFF_ENV = os.getenv('TRAIN_CUTOFF')
if TRAIN_CUTOFF_ENV:
    TRAIN_CUTOFF = pd.to_datetime(TRAIN_CUTOFF_ENV)
    if TRAIN_CUTOFF.tzinfo is None:
        TRAIN_CUTOFF = TRAIN_CUTOFF.tz_localize('Europe/Berlin')
    else:
        TRAIN_CUTOFF = TRAIN_CUTOFF.tz_convert('Europe/Berlin')
else:
    TRAIN_CUTOFF = pd.Timestamp('2023-12-31 23:45', tz='Europe/Berlin')

FEATURES = ['high', 'low', 'close', 'volume']
LAGS = [1, 2, 3, 4]
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


def run(max_assets=1000):
    # build asset list
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

    # We'll stream forecasts to disk as assets complete to avoid high memory usage
    forecasts = []
    results_rows = []
    out_csv = OUT_DIR / 'test_forecasts.csv'
    # write header
    pd.DataFrame(columns=['asset','feature','origin','horizon','target_time','pred','true']).to_csv(out_csv, index=False)
    def process_single_asset(p):
        aid = p.stem
        local_forecasts = []
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print('Failed reading', p, e)
            return local_forecasts, []

        # build lag features
        full = df.copy()
        for lag in LAGS:
            for f in FEATURES:
                full[f'f_lag{lag}'] = full[f].shift(lag)
        full['hour'] = full.index.hour
        full['dow'] = full.index.dayofweek

        # split
        df_trainpool = full.loc[full.index <= TRAIN_CUTOFF]
        df_test = full.loc[full.index > TRAIN_CUTOFF]
        if len(df_trainpool) < max(LAGS) + HORIZON + 10 or len(df_test) == 0:
            return local_forecasts, []

        # activity stats
        activity = {}
        for f in FEATURES:
            vals = df_trainpool[f].dropna().values
            if len(vals) == 0:
                activity[f] = {'mean_abs': 0.0, 'std': 0.0, 'nonzero_frac': 0.0}
            else:
                activity[f] = {
                    'mean_abs': float(np.mean(np.abs(vals))),
                    'std': float(np.std(vals)),
                    'nonzero_frac': float(np.mean(np.abs(vals) > CLS_EPS)),
                }

        feature_cols = [f'f_lag{lag}' for lag in LAGS] + ['hour', 'dow']

        for feat in FEATURES:
            is_active = (
                activity[feat]['mean_abs'] >= ACTIVITY_MEAN_THRESH
                or activity[feat]['std'] >= ACTIVITY_STD_THRESH
                or activity[feat]['nonzero_frac'] >= ACTIVITY_NONZERO_FRAC
            )

            models = {}
            clfs = {}

            for h in range(1, HORIZON + 1):
                tr = df_trainpool.copy()
                tr[f'y_h{h}'] = tr[feat].shift(-h)
                tr = tr.dropna(subset=[f'y_h{h}'])
                if len(tr) < 50 or not is_active:
                    models[h] = None
                    clfs[h] = None
                    continue

                X_tr = tr[feature_cols]
                y_tr = tr[f'y_h{h}']
                lgb_tr = lgb.Dataset(X_tr, y_tr)
                params = {
                    'objective': 'regression',
                    'metric': 'l2',
                    'verbosity': -1,
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'seed': 42,
                }
                bst = lgb.train(params, lgb_tr, num_boost_round=NUM_BOOST_ROUND)
                models[h] = bst

                if USE_CLASSIFIER:
                    y_bin = (np.abs(y_tr.values) > CLS_EPS).astype(int)
                    if 5 < y_bin.sum() < (len(y_bin) - 5):
                        try:
                            clf = LogisticRegression(max_iter=200, class_weight='balanced')
                            clf.fit(X_tr.fillna(0.0).values, y_bin)
                            clfs[h] = clf
                        except Exception:
                            clfs[h] = None
                    else:
                        clfs[h] = None
                else:
                    clfs[h] = None

            # batch predict for valid origins
            X_test_full = df_test[feature_cols].copy()
            valid_mask = ~X_test_full.isnull().any(axis=1)
            if not valid_mask.any():
                continue
            X_valid = X_test_full.loc[valid_mask]
            valid_origins = X_valid.index

            for h in range(1, HORIZON + 1):
                target_times = [orig + pd.Timedelta(minutes=15 * h) for orig in valid_origins]
                if not is_active:
                    preds = np.zeros(len(valid_origins), dtype=float)
                    proba = None
                else:
                    model = models.get(h)
                    if model is None:
                        preds = np.zeros(len(valid_origins), dtype=float)
                        proba = None
                    else:
                        preds = model.predict(
                            X_valid, num_iteration=getattr(model, 'best_iteration', None)
                        )
                        preds = np.asarray(preds, dtype=float)
                        clf = clfs.get(h)
                        if clf is not None:
                            try:
                                proba = clf.predict_proba(X_valid.fillna(0.0).values)[:, 1]
                            except Exception:
                                proba = None
                        else:
                            proba = None

                if proba is not None:
                    preds[proba < 0.5] = 0.0
                if PRED_THRESH is not None and PRED_THRESH > 0:
                    preds[np.abs(preds) < PRED_THRESH] = 0.0

                for i, origin in enumerate(valid_origins):
                    target_time = target_times[i]
                    true_val = None
                    if target_time in df.index:
                        true_val = df.loc[target_time, feat]
                    local_forecasts.append(
                        {
                            'asset': aid,
                            'feature': feat,
                            'origin': origin,
                            'horizon': h,
                            'target_time': target_time,
                            'pred': float(preds[i]),
                            'true': (float(true_val) if true_val is not None else None),
                        }
                    )

        return local_forecasts, []


def process_asset_worker(p):
    """Top-level worker function used by multiprocessing.Pool.
    This mirrors the nested process_single_asset logic but is picklable.
    """
    aid = p.stem
    local_forecasts = []
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        print('Failed reading', p, e)
        return local_forecasts, []

    # build lag features
    full = df.copy()
    for lag in LAGS:
        for f in FEATURES:
            full[f'f_lag{lag}'] = full[f].shift(lag)
    full['hour'] = full.index.hour
    full['dow'] = full.index.dayofweek

    # split
    df_trainpool = full.loc[full.index <= TRAIN_CUTOFF]
    df_test = full.loc[full.index > TRAIN_CUTOFF]
    if len(df_trainpool) < max(LAGS) + HORIZON + 10 or len(df_test) == 0:
        return local_forecasts, []

    # activity stats
    activity = {}
    for f in FEATURES:
        vals = df_trainpool[f].dropna().values
        if len(vals) == 0:
            activity[f] = {'mean_abs': 0.0, 'std': 0.0, 'nonzero_frac': 0.0}
        else:
            activity[f] = {
                'mean_abs': float(np.mean(np.abs(vals))),
                'std': float(np.std(vals)),
                'nonzero_frac': float(np.mean(np.abs(vals) > CLS_EPS)),
            }

    feature_cols = [f'f_lag{lag}' for lag in LAGS] + ['hour', 'dow']

    for feat in FEATURES:
        is_active = (
            activity[feat]['mean_abs'] >= ACTIVITY_MEAN_THRESH
            or activity[feat]['std'] >= ACTIVITY_STD_THRESH
            or activity[feat]['nonzero_frac'] >= ACTIVITY_NONZERO_FRAC
        )

        models = {}
        clfs = {}

        for h in range(1, HORIZON + 1):
            tr = df_trainpool.copy()
            tr[f'y_h{h}'] = tr[feat].shift(-h)
            tr = tr.dropna(subset=[f'y_h{h}'])
            if len(tr) < 50 or not is_active:
                models[h] = None
                clfs[h] = None
                continue

            X_tr = tr[feature_cols]
            y_tr = tr[f'y_h{h}']
            lgb_tr = lgb.Dataset(X_tr, y_tr)
            params = {
                'objective': 'regression',
                'metric': 'l2',
                'verbosity': -1,
                'num_leaves': 31,
                'learning_rate': 0.05,
                'seed': 42,
            }
            bst = lgb.train(params, lgb_tr, num_boost_round=NUM_BOOST_ROUND)
            models[h] = bst

            if USE_CLASSIFIER:
                y_bin = (np.abs(y_tr.values) > CLS_EPS).astype(int)
                if 5 < y_bin.sum() < (len(y_bin) - 5):
                    try:
                        clf = LogisticRegression(max_iter=200, class_weight='balanced')
                        clf.fit(X_tr.fillna(0.0).values, y_bin)
                        clfs[h] = clf
                    except Exception:
                        clfs[h] = None
                else:
                    clfs[h] = None
            else:
                clfs[h] = None

        # batch predict for valid origins
        X_test_full = df_test[feature_cols].copy()
        valid_mask = ~X_test_full.isnull().any(axis=1)
        if not valid_mask.any():
            return local_forecasts, []
        X_valid = X_test_full.loc[valid_mask]
        valid_origins = X_valid.index

        for h in range(1, HORIZON + 1):
            target_times = [orig + pd.Timedelta(minutes=15 * h) for orig in valid_origins]
            if not is_active:
                preds = np.zeros(len(valid_origins), dtype=float)
                proba = None
            else:
                model = models.get(h)
                if model is None:
                    preds = np.zeros(len(valid_origins), dtype=float)
                    proba = None
                else:
                    preds = model.predict(
                        X_valid, num_iteration=getattr(model, 'best_iteration', None)
                    )
                    preds = np.asarray(preds, dtype=float)
                    clf = clfs.get(h)
                    if clf is not None:
                        try:
                            proba = clf.predict_proba(X_valid.fillna(0.0).values)[:, 1]
                        except Exception:
                            proba = None
                    else:
                        proba = None

            if proba is not None:
                preds[proba < 0.5] = 0.0
            if PRED_THRESH is not None and PRED_THRESH > 0:
                preds[np.abs(preds) < PRED_THRESH] = 0.0

            for i, origin in enumerate(valid_origins):
                target_time = target_times[i]
                true_val = None
                if target_time in df.index:
                    true_val = df.loc[target_time, feat]
                local_forecasts.append(
                    {
                        'asset': aid,
                        'feature': feat,
                        'origin': origin,
                        'horizon': h,
                        'target_time': target_time,
                        'pred': float(preds[i]),
                        'true': (float(true_val) if true_val is not None else None),
                    }
                )

    return local_forecasts, []

    # run pool (call module-level worker to avoid pickling local functions)
    n_workers = int(os.getenv('N_WORKERS', max(1, mp.cpu_count() - 1)))
    with mp.Pool(processes=n_workers) as pool:
        for fc_list in pool.imap_unordered(process_asset_worker, assets, chunksize=1):
            if fc_list:
                # append to CSV to avoid storing all forecasts in memory
                try:
                    pd.DataFrame(fc_list).to_csv(out_csv, mode='a', header=False, index=False)
                except Exception as e:
                    print('Failed to append forecasts for an asset:', e)

    # save forecasts
    outp = OUT_DIR / 'test_forecasts.parquet'
    df_fc = pd.DataFrame(forecasts)
    if not df_fc.empty:
        df_fc['origin'] = pd.to_datetime(df_fc['origin'])
        df_fc['target_time'] = pd.to_datetime(df_fc['target_time'])
        df_fc.to_parquet(outp, index=False)
        print('Wrote forecasts to', outp)

        # compute sMAPE per asset/feature/horizon
        eval_rows = []
        grouped = df_fc.dropna(subset=['true']).groupby(['asset', 'feature', 'horizon'])
        for (asset, feat, h), g in grouped:
            score = smape(g['true'].values, g['pred'].values)
            eval_rows.append({'asset': asset, 'feature': feat, 'horizon': h, 'sMAPE': score, 'n': len(g)})
        out_eval = OUT_DIR / 'per_asset_test_results.csv'
        pd.DataFrame(eval_rows).to_csv(out_eval, index=False)
        print('Wrote per-asset test results to', out_eval)
    else:
        print('No forecasts generated')


if __name__ == '__main__':
    max_assets = int(os.getenv('MAX_ASSETS', 1000))
    run(max_assets=max_assets)
#!/usr/bin/env python3
"""
Retrain per-asset models on full TRAIN (up to TRAIN_CUTOFF) and forecast for TEST (after TRAIN_CUTOFF).
Writes:
 - experiments/test_forecasts.parquet: columns [asset, feature, origin, horizon, target_time, pred, true]
 - experiments/per_asset_test_results.csv: aggregated sMAPE per asset/feature/horizon
"""
from pathlib import Path
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
import multiprocessing as mp
from functools import partial

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / 'data' / 'assets_parquet'
OUT_DIR = ROOT / 'experiments'
OUT_DIR.mkdir(exist_ok=True)

HORIZON = int(os.getenv('HORIZON', 10))
NUM_BOOST_ROUND = int(os.getenv('NUM_BOOST_ROUND', 100))
TRAIN_CUTOFF_ENV = os.getenv('TRAIN_CUTOFF')
if TRAIN_CUTOFF_ENV:
    TRAIN_CUTOFF = pd.to_datetime(TRAIN_CUTOFF_ENV)
    if TRAIN_CUTOFF.tzinfo is None:
        TRAIN_CUTOFF = TRAIN_CUTOFF.tz_localize('Europe/Berlin')
    else:
        TRAIN_CUTOFF = TRAIN_CUTOFF.tz_convert('Europe/Berlin')
else:
    TRAIN_CUTOFF = pd.Timestamp('2023-12-31 23:45', tz='Europe/Berlin')

FEATURES = ['high', 'low', 'close', 'volume']
LAGS = [1,2,3,4]
PRED_THRESH = float(os.getenv('PRED_THRESH', 1e-3))
ACTIVITY_MEAN_THRESH = float(os.getenv('ACTIVITY_MEAN_THRESH', 0.01))
ACTIVITY_STD_THRESH = float(os.getenv('ACTIVITY_STD_THRESH', 0.01))
ACTIVITY_NONZERO_FRAC = float(os.getenv('ACTIVITY_NONZERO_FRAC', 0.05))
USE_CLASSIFIER = int(os.getenv('USE_CLASSIFIER', 1))
CLS_EPS = float(os.getenv('CLS_EPS', 1e-6))
MAX_ASSETS = int(os.getenv('MAX_ASSETS', 1000))


def smape(y_true, y_pred, eps=1e-8):
    num = np.abs(y_pred - y_true)
    den = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    res = np.zeros_like(num, dtype=float)
    mask = den > eps
    if np.any(mask):
        res[mask] = num[mask] / den[mask]
    return np.nanmean(res) * 100.0


def run(max_assets=1000):
    # optional asset list
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

    forecasts = []
    results_rows = []

    # Use multiprocessing to process assets in parallel to speed up execution
    def process_single_asset(p):
        aid = p.stem
        local_forecasts = []
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            print('Failed reading', p, e)
            return local_forecasts, []

        # build lag features
        full = df.copy()
        for lag in LAGS:
            for f in FEATURES:
                full[f'f_lag{lag}'] = full[f].shift(lag)
        full['hour'] = full.index.hour
        full['dow'] = full.index.dayofweek

        # train on up to TRAIN_CUTOFF
        df_trainpool = full.loc[full.index <= TRAIN_CUTOFF]
        df_test = full.loc[full.index > TRAIN_CUTOFF]
        if len(df_trainpool) < max(LAGS) + HORIZON + 10 or len(df_test) == 0:
            # not enough data
            return local_forecasts, []

        # compute activity stats
        activity = {}
        for f in FEATURES:
            vals = df_trainpool[f].dropna().values
            if len(vals) == 0:
                activity[f] = {'mean_abs':0.0, 'std':0.0, 'nonzero_frac':0.0}
            else:
                activity[f] = {
                    'mean_abs': float(np.mean(np.abs(vals))),
                    'std': float(np.std(vals)),
                    'nonzero_frac': float(np.mean(np.abs(vals) > CLS_EPS))
                }

        feature_cols = [f'f_lag{lag}' for lag in LAGS] + ['hour','dow']
        eval_rows_local = []

        for feat in FEATURES:
            is_active = (activity[feat]['mean_abs'] >= ACTIVITY_MEAN_THRESH) or (activity[feat]['std'] >= ACTIVITY_STD_THRESH) or (activity[feat]['nonzero_frac'] >= ACTIVITY_NONZERO_FRAC)
            models = {}
            clfs = {}
            for h in range(1, HORIZON+1):
                tr = df_trainpool.copy()
                tr[f'y_h{h}'] = tr[feat].shift(-h)
                tr = tr.dropna(subset=[f'y_h{h}'])
                if len(tr) < 50 or not is_active:
                    models[h] = None
                    clfs[h] = None
                    continue
                X_tr = tr[feature_cols]
                y_tr = tr[f'y_h{h}']
                lgb_tr = lgb.Dataset(X_tr, y_tr)
                params = {'objective':'regression', 'metric':'l2', 'verbosity':-1, 'num_leaves':31, 'learning_rate':0.05, 'seed':42}
                bst = lgb.train(params, lgb_tr, num_boost_round=NUM_BOOST_ROUND)
                models[h] = bst

                if USE_CLASSIFIER:
                    y_bin = (np.abs(y_tr.values) > CLS_EPS).astype(int)
                    if y_bin.sum() > 5 and (y_bin.sum() < len(y_bin) - 5):
                        try:
                            clf = LogisticRegression(max_iter=200, class_weight='balanced')
                            clf.fit(X_tr.fillna(0.0).values, y_bin)
                            clfs[h] = clf
                        except Exception:
                            clfs[h] = None
                    else:
                        clfs[h] = None
                else:
                    clfs[h] = None

            # batch predict
            X_test_full = df_test[feature_cols].copy()
            valid_mask = ~X_test_full.isnull().any(axis=1)
            if not valid_mask.any():
                continue
            X_valid = X_test_full.loc[valid_mask]
            valid_origins = X_valid.index

            for h in range(1, HORIZON+1):
                target_times = [orig + pd.Timedelta(minutes=15*h) for orig in valid_origins]
                if not is_active:
                    preds = np.zeros(len(valid_origins), dtype=float)
                    proba = None
                else:
                    model = models.get(h)
                    if model is None:
                        preds = np.zeros(len(valid_origins), dtype=float)
                        proba = None
                    else:
                        preds = model.predict(X_valid, num_iteration=model.best_iteration if hasattr(model, 'best_iteration') else None)
                        preds = np.asarray(preds, dtype=float)
                        clf = clfs.get(h)
                        if clf is not None:
                            try:
                                proba = clf.predict_proba(X_valid.fillna(0.0).values)[:, 1]
                            except Exception:
                                proba = None
                        else:
                            proba = None

                if proba is not None:
                    mask_low = proba < 0.5
                    preds[mask_low] = 0.0
                if PRED_THRESH is not None and PRED_THRESH > 0:
                    small_mask = np.abs(preds) < PRED_THRESH
                    preds[small_mask] = 0.0

                for i, origin in enumerate(valid_origins):
                    target_time = target_times[i]
                    true_val = None
                    if target_time in df.index:
                        true_val = df.loc[target_time, feat]
                    local_forecasts.append({'asset': aid, 'feature': feat, 'origin': origin, 'horizon': h, 'target_time': target_time, 'pred': float(preds[i]), 'true': (float(true_val) if true_val is not None else None)})

        return local_forecasts, eval_rows_local

    # prepare pool
    n_workers = int(os.getenv('N_WORKERS', max(1, mp.cpu_count() - 1)))
    with mp.Pool(processes=n_workers) as pool:
        for fc_list, _ in pool.imap_unordered(process_asset_worker, assets, chunksize=1):
            if fc_list:
                forecasts.extend(fc_list)

    # save forecasts
    outp = OUT_DIR / 'test_forecasts.parquet'
    df_fc = pd.DataFrame(forecasts)
    if not df_fc.empty:
        # ensure origin and target_time are timestamps
        df_fc['origin'] = pd.to_datetime(df_fc['origin'])
        df_fc['target_time'] = pd.to_datetime(df_fc['target_time'])
        df_fc.to_parquet(outp, index=False)
        print('Wrote forecasts to', outp)

        # compute sMAPE per asset/feature/horizon
        eval_rows = []
        grouped = df_fc.dropna(subset=['true']).groupby(['asset','feature','horizon'])
        for (asset,feat,h), g in grouped:
            score = smape(g['true'].values, g['pred'].values)
            eval_rows.append({'asset':asset,'feature':feat,'horizon':h,'sMAPE':score,'n':len(g)})
        out_eval = OUT_DIR / 'per_asset_test_results.csv'
        pd.DataFrame(eval_rows).to_csv(out_eval, index=False)
        print('Wrote per-asset test results to', out_eval)
    else:
        print('No forecasts generated')


if __name__ == '__main__':
    max_assets = int(os.getenv('MAX_ASSETS', 1000))
    run(max_assets=max_assets)
