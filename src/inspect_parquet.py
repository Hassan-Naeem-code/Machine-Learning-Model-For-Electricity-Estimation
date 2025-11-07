#!/usr/bin/env python3
"""
Quick inspect script for the downloaded parquet files.

Print schema, number of rows (approx) and a small sample head.
"""
import os
import sys

from pathlib import Path

TRAIN = Path(__file__).resolve().parents[1] / 'data' / 'TRAIN_Reco_2021_2022_2023.parquet.gzip'
TEST = Path(__file__).resolve().parents[1] / 'data' / 'TEST_Reco_2024.parquet.gzip'

try:
    import pyarrow.parquet as pq
except Exception as e:
    print('ERROR: pyarrow not available:', e)
    print('Install dependencies: python -m pip install -r ../requirements.txt')
    sys.exit(1)


def inspect(path):
    print('Inspecting', path)
    t = pq.ParquetFile(str(path))
    print('Num row groups:', t.num_row_groups)
    try:
        meta = t.metadata
        print('Num rows (metadata):', meta.num_rows)
    except Exception:
        pass
    print('Schema:')
    print(t.schema)
    print('\nSample rows:')
    # read first row group or first 1000 rows
    try:
        table = t.read_row_group(0, columns=None)
        df = table.to_pandas()
        print(df.head().to_string())
    except Exception as e:
        print('Failed reading row group, trying slice read:', e)
        try:
            df = t.read(columns=None).to_pandas().head()
            print(df.head().to_string())
        except Exception as e2:
            print('Failed reading file:', e2)


if __name__ == '__main__':
    for p in (TRAIN, TEST):
        if p.exists():
            inspect(p)
        else:
            print('Missing', p)
