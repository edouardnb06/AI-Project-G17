"""Supervised windows for several horizons (advanced, optional).

Cuts a wide run (see src/load.py) into inputs and targets for several horizons
at once. The beginner task uses the ready-made tables in data/tables/ instead.
"""
import numpy as np
import pandas as pd

from src.load import DATA_DIR, FEATURES, TARGET, load_run, load_splits, to_wide

HORIZONS = (1, 5, 10, 25)    # samples ahead: 2, 10, 20 and 50 ms


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
