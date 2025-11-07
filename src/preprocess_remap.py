#!/usr/bin/env python3
"""
Preprocess the raw parquet files into per-asset, regularly-indexed parquet files.

Steps:
- Iterate row-groups from TRAIN and TEST parquet files
- For each row-group, write rows grouped by `ID` into per-asset CSV (append)
- After collecting all rows, for each asset read its CSV, parse timestamps, create a full 15-min index
  covering the full range observed, align data, fill missing with zeros and save parquet to
  `data/assets_parquet/{ID}.parquet`.

This approach avoids loading the full dataset into memory.
"""
import os
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
TRAIN = DATA / 'TRAIN_Reco_2021_2022_2023.parquet.gzip'
TEST = DATA / 'TEST_Reco_2024.parquet.gzip'
TMP_DIR = DATA / 'raw_by_asset_csv'
OUT_DIR = DATA / 'assets_parquet'

os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

files = [TRAIN, TEST]

print('Phase 1: split by asset into CSV files (append mode)')
for f in files:
    if not f.exists():
        print('Missing', f)
        continue
    print('Processing', f.name)
    pqf = pq.ParquetFile(str(f))
    for rg in range(pqf.num_row_groups):
        table = pqf.read_row_group(rg)
        df = table.to_pandas()
        # If ExecutionTime is the index, reset it to a column
        if df.index.name == 'ExecutionTime':
            df = df.reset_index()
        # ensure ExecutionTime column exists
        if 'ExecutionTime' in df.columns:
            # convert to datetime but keep possible tz info
            df['ExecutionTime'] = pd.to_datetime(df['ExecutionTime'], utc=True, errors='coerce')
        else:
            df['ExecutionTime'] = pd.NaT
        # group by ID and append to CSV
        for id_val, g in df.groupby('ID'):
            outp = TMP_DIR / f"{id_val}.csv"
            header = not outp.exists()
            # write only needed columns
            g[['ExecutionTime','high','low','close','volume']].to_csv(outp, mode='a', header=header, index=False)

print('\nPhase 2: create regular 15-min indexed parquet per asset')

# Determine global full range from all asset CSVs by scanning min/max
global_min = None
global_max = None
for csvf in TMP_DIR.glob('*.csv'):
    df0 = pd.read_csv(csvf, usecols=['ExecutionTime'])
    # parse and coerce; some rows may be empty
    df0['ExecutionTime'] = pd.to_datetime(df0['ExecutionTime'], utc=True, errors='coerce')
    if df0['ExecutionTime'].notna().any():
        lo = df0['ExecutionTime'].min()
        hi = df0['ExecutionTime'].max()
        if global_min is None or lo < global_min:
            global_min = lo
        if global_max is None or hi > global_max:
            global_max = hi

if global_min is None:
    raise SystemExit('No timestamps found in CSVs')

# normalize timezone: convert to Europe/Berlin (handles DST)
if global_min.tz is None:
    global_min = global_min.tz_localize('UTC').tz_convert('Europe/Berlin')
else:
    global_min = global_min.tz_convert('Europe/Berlin')
if global_max.tz is None:
    global_max = global_max.tz_localize('UTC').tz_convert('Europe/Berlin')
else:
    global_max = global_max.tz_convert('Europe/Berlin')

full_index = pd.date_range(start=global_min.floor('15T'), end=global_max.ceil('15T'), freq='15T', tz='Europe/Berlin')
print('Full index from', full_index[0], 'to', full_index[-1], 'len=', len(full_index))

for csvf in sorted(TMP_DIR.glob('*.csv')):
    id_val = csvf.stem
    try:
        df = pd.read_csv(csvf, parse_dates=['ExecutionTime'])
    except Exception:
        # fallback: read without parse and convert
        df = pd.read_csv(csvf)
        df['ExecutionTime'] = pd.to_datetime(df['ExecutionTime'])
    # localize/convert timezone like before
    if df['ExecutionTime'].dt.tz is None:
        # assume UTC then convert
        df['ExecutionTime'] = df['ExecutionTime'].dt.tz_localize('UTC').dt.tz_convert('Europe/Berlin')
    else:
        df['ExecutionTime'] = df['ExecutionTime'].dt.tz_convert('Europe/Berlin')
    df = df.set_index('ExecutionTime')
    # aggregate duplicates by taking last (shouldn't be many)
    df = df.groupby(df.index).last()
    # reindex to full_index and fill zeros where missing
    df = df.reindex(full_index)
    df[['high','low','close','volume']] = df[['high','low','close','volume']].fillna(0.0)
    outp = OUT_DIR / f"{id_val}.parquet"
    df.to_parquet(outp)
    print('Wrote', outp, 'rows=', len(df))

print('Done. Per-asset parquet files in', OUT_DIR)
