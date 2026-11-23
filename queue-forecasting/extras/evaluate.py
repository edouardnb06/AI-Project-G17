"""Metrics per horizon: MAE, RMSE, pinball loss and 80% interval coverage.

Shapes: y and point forecasts are (n, H), one column per horizon; quantile
forecasts are (n, H, 3) for the quantiles 0.1, 0.5 and 0.9. Targets that are
NaN (lost samples, end of run) are left out of every metric.
"""
import numpy as np
import pandas as pd

QUANTILES = (0.1, 0.5, 0.9)


def _masked(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return np.ma.masked_invalid(y), pred


def mae(y, point):
    y, point = _masked(y, point)
    return np.ma.filled(np.abs(y - point).mean(axis=0), np.nan)


def rmse(y, point):
    y, point = _masked(y, point)
    return np.ma.filled(np.sqrt(((y - point) ** 2).mean(axis=0)), np.nan)


def pinball(y, q, tau):
    """Mean pinball (quantile) loss of the tau-quantile forecast q."""
    y, q = _masked(y, q)
    diff = y - q
    return np.ma.filled(np.maximum(tau * diff, (tau - 1) * diff).mean(axis=0), np.nan)


def coverage(y, lo, hi):
    """Share of targets inside [lo, hi]; 0.8 is ideal for the 0.1-0.9 interval."""
    y, lo = _masked(y, lo)
    inside = (y >= lo) & (y <= np.asarray(hi, dtype=float))
    return np.ma.filled(inside.mean(axis=0), np.nan)


def evaluate(y, point, quant, horizons, sample_ms=2.0):
    """One row per horizon: MAE, RMSE, pinball per quantile and mean, coverage_80."""
    table = pd.DataFrame(index=pd.Index(horizons, name="horizon"))
    table["ms"] = [h * sample_ms for h in horizons]
    table["n"] = np.isfinite(np.asarray(y, dtype=float)).sum(axis=0)
    table["MAE"] = mae(y, point)
    table["RMSE"] = rmse(y, point)
    for i, tau in enumerate(QUANTILES):
        table[f"pinball_{tau}"] = pinball(y, quant[..., i], tau)
    table["pinball"] = table[[f"pinball_{tau}" for tau in QUANTILES]].mean(axis=1)
    table["coverage_80"] = coverage(y, quant[..., 0], quant[..., -1])
    return table
