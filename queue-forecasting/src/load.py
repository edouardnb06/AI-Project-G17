"""Load runs and splits, and cut runs into supervised windows.

A run is stored in long format: one row per (seq, hop). For forecasting it is
easier to work in wide format: one row per seq, with the readings of both
switches side by side and a NaN row for every lost sample (see to_wide).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

SAMPLE_MS = 2.0              # one sample every 2 ms
SAMPLES_PER_RUN = 45_000     # 90 s per run
BUFFER_PKTS = 64             # bottleneck buffer size
HORIZONS = (1, 5, 10, 25)    # samples ahead: 2, 10, 20 and 50 ms

TARGET = "qdepth_pkts_h0"    # bottleneck queue depth
FEATURES = ("qdepth_pkts_h0", "qdelay_us_h0", "qdepth_pkts_h1", "qdelay_us_h1")


def list_runs(data_dir=DATA_DIR):
    """Run ids found in data/runs, sorted."""
    return sorted(p.stem for p in (Path(data_dir) / "runs").glob("run_*.parquet"))


def load_run(run_id, data_dir=DATA_DIR):
    """One run in long format, as stored (one row per seq and hop)."""
    return pd.read_parquet(Path(data_dir) / "runs" / f"{run_id}.parquet")


def load_splits(data_dir=DATA_DIR):
    """{'train': [...], 'validation': [...], 'test': [...]} run ids."""
    splits = json.loads((Path(data_dir) / "splits.json").read_text())
    return {k: splits[k] for k in ("train", "validation", "test")}


def to_wide(run):
    """One row per seq from 0 to the last received seq.

    Columns: t_ms, qdepth_pkts_h0, qdelay_us_h0, qdepth_pkts_h1, qdelay_us_h1
    (suffix _h0 = bottleneck switch, _h1 = downstream switch). Lost samples
    are rows of NaN, so row i is always sample i.
    """
    wide = run.pivot(index="seq", columns="hop", values=["qdepth_pkts", "qdelay_us"])
    wide.columns = [f"{name}_h{hop}" for name, hop in wide.columns]
    wide = wide[list(FEATURES)].astype("float64")
    wide.insert(0, "t_ms", run.groupby("seq")["t_ms"].first())
    return wide.reindex(pd.RangeIndex(0, int(run["seq"].max()) + 1, name="seq"))


def load_wide(run_ids, data_dir=DATA_DIR):
    """{run_id: wide frame} for a list of run ids."""
    return {r: to_wide(load_run(r, data_dir)) for r in run_ids}


def make_windows(wide, lookback=50, horizons=HORIZONS, features=FEATURES, stride=1):
    """Supervised windows from one wide run.

    Forecast origins are the samples t where the bottleneck depth was received
    and a full lookback fits (t >= lookback - 1); keep every stride-th one.

    Returns
      X      float32 (n, lookback, len(features)): samples t-lookback+1 .. t,
             NaN where a sample was lost (filling is left to the model)
      Y      float32 (n, len(horizons)): bottleneck depth at t + h,
             NaN where that sample was lost or lies past the end of the run
      origin int64 (n,): seq of each origin t
    """
    values = wide[list(features)].to_numpy(dtype=np.float32)
    target = wide[TARGET].to_numpy(dtype=np.float32)
    n = len(wide)
    origin = np.arange(lookback - 1, n)
    origin = origin[~np.isnan(target[origin])][::stride]

    view = np.lib.stride_tricks.sliding_window_view(values, lookback, axis=0)
    X = np.ascontiguousarray(view[origin - lookback + 1].transpose(0, 2, 1))

    padded = np.concatenate([target, np.full(max(horizons), np.nan, np.float32)])
    Y = np.stack([padded[origin + h] for h in horizons], axis=1)
    return X, Y, origin


def load_windows(split, lookback=50, horizons=HORIZONS, features=FEATURES, stride=1,
                 data_dir=DATA_DIR):
    """Windows for all runs of a split ('train', 'validation' or 'test').

    Returns X, Y and a DataFrame (run_id, seq) giving the origin of each row,
    which is what grouped cross-validation by run needs.
    """
    xs, ys, meta = [], [], []
    for run_id, wide in load_wide(load_splits(data_dir)[split], data_dir).items():
        X, Y, origin = make_windows(wide, lookback, horizons, features, stride)
        xs.append(X)
        ys.append(Y)
        meta.append(pd.DataFrame({"run_id": run_id, "seq": origin}))
    return np.concatenate(xs), np.concatenate(ys), pd.concat(meta, ignore_index=True)
