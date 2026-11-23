"""Build the ready-made tables in data/tables/ from the raw runs in data/runs/.

You only need this if you want to rebuild the tables. Run it from the
queue-forecasting folder:

    python -m src.make_tables          # every 5th row: the tables in data/tables/
    python -m src.make_tables --all    # every row: larger tables in data/tables_all/

One row of a table is one moment t of one run. All features come from the
samples up to t; only the target y looks ahead.
"""
import argparse

import numpy as np
import pandas as pd

from src import load

TABLES_DIR = load.DATA_DIR / "tables"           # every 5th row, shared with the group
TABLES_ALL_DIR = load.DATA_DIR / "tables_all"   # every row, not shared (too large)

HORIZON = 10     # y is the queue length 10 samples (20 ms) after now
LAGS = 10        # lag_1 ... lag_10
HISTORY = 50     # mean_50 and max_50 need the last 50 samples
EVERY = 5        # keep every 5th row

SPLITS = {"train": "train", "val": "validation", "test": "test"}  # file name -> name in splits.json


def make_table(run, every=EVERY):
    """Turn one raw run into table rows, keeping every `every`-th row."""
    wide = load.to_wide(run)            # one row per sample; lost samples are NaN
    queue = wide[load.TARGET]

    table = pd.DataFrame({"run_id": run["run_id"].iloc[0], "t_ms": wide["t_ms"]})
    table["y"] = queue.shift(-HORIZON)
    for k in range(1, LAGS + 1):
        table[f"lag_{k}"] = queue.shift(k - 1)      # lag_1 is the queue now
    table["mean_10"] = queue.rolling(10).mean()
    table["mean_50"] = queue.rolling(HISTORY).mean()
    table["max_50"] = queue.rolling(HISTORY).max()
    table["slope_10"] = slope_10(queue)
    table["delay_now_ms"] = wide["qdelay_us_h0"] / 1000

    # A row covers the 50 samples of its history and the 10 samples up to its
    # target. Drop it if any of these 60 samples is lost or outside the run.
    span = HISTORY + HORIZON
    received = queue.notna().astype(int)
    complete = received.rolling(span).sum().shift(-HORIZON) == span
    table = table[complete].iloc[::every].reset_index(drop=True)

    whole_numbers = ["y", "max_50"] + [f"lag_{k}" for k in range(1, LAGS + 1)]
    return table.astype({column: "int64" for column in whole_numbers})


def slope_10(queue):
    """Slope of the straight line fitted through the last 10 samples (packets per sample)."""
    x = np.arange(10)                               # 0 = oldest, 9 = now
    weights = (x - x.mean()) / ((x - x.mean()) ** 2).sum()
    return sum(weight * queue.shift(9 - i) for i, weight in enumerate(weights))


def build(every=EVERY, out_dir=TABLES_DIR):
    """Write train.parquet, val.parquet and test.parquet to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = load.load_splits()
    for name, split in SPLITS.items():
        table = pd.concat([make_table(load.load_run(r), every) for r in splits[split]],
                          ignore_index=True)
        path = out_dir / f"{name}.parquet"
        table.to_parquet(path, index=False, compression="zstd")
        print(f"{path.name}: {len(table):,} rows, {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the tables in data/tables/.")
    parser.add_argument("--all", action="store_true",
                        help="keep every row and write to data/tables_all/")
    if parser.parse_args().all:
        build(every=1, out_dir=TABLES_ALL_DIR)
    else:
        build()
