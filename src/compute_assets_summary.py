#!/usr/bin/env python3
"""
Compute per-asset summary statistics from the TRAIN and TEST parquet files.
Saves `data/assets_summary.csv` with columns:
- ID, total_count, nonzero_count, fraction_zero, first_ts, last_ts

The script iterates row-groups to avoid loading entire files into memory.
"""
import os
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
TRAIN = DATA / 'TRAIN_Reco_2021_2022_2023.parquet.gzip'
TEST = DATA / 'TEST_Reco_2024.parquet.gzip'
OUT = DATA / 'assets_summary.csv'

files = [TRAIN, TEST]

stats = {}

for f in files:
    if not f.exists():
        print('Missing', f)
        continue
    print('Processing', f.name)
    pqf = pq.ParquetFile(str(f))
    for rg in range(pqf.num_row_groups):
        try:
            table = pqf.read_row_group(rg, columns=['ID','high','low','close','volume','ExecutionTime'])
        except Exception as e:
            # fallback to reading whole row group without ExecutionTime
            print('Read row group failed:', e)
            table = pqf.read_row_group(rg)
        df = table.to_pandas()
        # Ensure ExecutionTime is datetime
        if 'ExecutionTime' in df.columns:
            df['ExecutionTime'] = pd.to_datetime(df['ExecutionTime'])
        else:
            df['ExecutionTime'] = pd.NaT
        # boolean zero row
        zero_mask = (df['high']==0) & (df['low']==0) & (df['close']==0) & (df['volume']==0)
        for id_val, g in df.groupby('ID'):
            cnt = len(g)
            nz = int((~zero_mask.loc[g.index]).sum())
            first = g['ExecutionTime'].min()
            last = g['ExecutionTime'].max()
            st = stats.get(id_val)
            if st is None:
                stats[id_val] = {
                    'total_count': cnt,
                    'nonzero_count': nz,
                    'first_ts': first,
                    'last_ts': last,
                }
            else:
                st['total_count'] += cnt
                st['nonzero_count'] += nz
                if pd.notna(first) and (pd.isna(st['first_ts']) or first < st['first_ts']):
                    st['first_ts'] = first
                if pd.notna(last) and (pd.isna(st['last_ts']) or last > st['last_ts']):
                    st['last_ts'] = last

# Build dataframe
rows = []
for id_val, v in stats.items():
    total = v['total_count']
    nonzero = v['nonzero_count']
    frac_zero = 1.0 - (nonzero / total) if total>0 else 1.0
    rows.append({
        'ID': id_val,
        'total_count': total,
        'nonzero_count': nonzero,
        'fraction_zero': frac_zero,
        'first_ts': v['first_ts'],
        'last_ts': v['last_ts'],
    })

df_out = pd.DataFrame(rows).sort_values('ID')
# Save
OUT.parent.mkdir(parents=True, exist_ok=True)
df_out.to_csv(OUT, index=False)
print('Wrote', OUT, 'with', len(df_out), 'assets')
