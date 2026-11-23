"""Supervised windows and metrics on small hand-made inputs."""
import numpy as np
import pandas as pd
import pytest

from extras import evaluate, windows
from src import load


def toy_run():
    """Six samples, sample 3 lost; bottleneck depth equals 10 * seq."""
    rows = []
    for seq in [0, 1, 2, 4, 5]:
        rows.append(("run_x", seq, 2.0 * seq, 0, 10 * seq, 100 * seq))
        rows.append(("run_x", seq, 2.0 * seq, 1, 0, 50))
    return pd.DataFrame(rows, columns=["run_id", "seq", "t_ms", "hop", "qdepth_pkts", "qdelay_us"])


def test_to_wide_keeps_lost_samples_as_nan_rows():
    wide = load.to_wide(toy_run())
    assert list(wide.index) == [0, 1, 2, 3, 4, 5]
    assert np.isnan(wide.loc[3, load.TARGET])
    assert wide.loc[4, load.TARGET] == 40


def test_windows_align_inputs_and_targets():
    wide = load.to_wide(toy_run())
    X, Y, origin = windows.make_windows(wide, lookback=2, horizons=(1, 2), features=(load.TARGET,))
    # origins are observed samples with a full lookback: seq 1, 2, 4, 5
    assert list(origin) == [1, 2, 4, 5]
    assert X.shape == (4, 2, 1) and Y.shape == (4, 2)
    assert list(X[0, :, 0]) == [0, 10]          # samples 0 and 1
    assert Y[0, 0] == 20 and np.isnan(Y[0, 1])  # sample 3 was lost
    assert np.isnan(X[2, 0, 0]) and X[2, 1, 0] == 40
    assert np.isnan(Y[3]).all()                 # past the end of the run


def test_point_metrics():
    y = np.array([[0.0], [2.0], [np.nan]])
    yhat = np.array([[1.0], [0.0], [5.0]])
    assert evaluate.mae(y, yhat)[0] == pytest.approx(1.5)
    assert evaluate.rmse(y, yhat)[0] == pytest.approx(np.sqrt(2.5))


def test_pinball_and_coverage():
    y = np.array([[0.0], [10.0]])
    q = np.array([[2.0], [2.0]])
    # tau = 0.9: under-forecast costs 0.9 per unit, over-forecast 0.1
    assert evaluate.pinball(y, q, 0.9)[0] == pytest.approx((0.1 * 2 + 0.9 * 8) / 2)
    lo, hi = np.array([[0.0], [0.0]]), np.array([[1.0], [5.0]])
    assert evaluate.coverage(y, lo, hi)[0] == pytest.approx(0.5)


def test_evaluate_table_shape():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(50, 4))
    point = np.zeros_like(y)
    quant = np.stack([point - 1, point, point + 1], axis=-1)
    table = evaluate.evaluate(y, point, quant, horizons=(1, 5, 10, 25))
    assert list(table.index) == [1, 5, 10, 25]
    assert {"MAE", "RMSE", "pinball", "coverage_80"} <= set(table.columns)
