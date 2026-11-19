"""The baselines run end to end on a few real runs."""
import numpy as np

from src import baselines, load


def test_baselines_end_to_end():
    train, valid = ["run_01", "run_02"], ["run_08"]
    table = baselines.run(train, valid)
    assert set(table["model"]) == {"persistence", "ewma"}
    assert sorted(table["horizon"].unique()) == list(load.HORIZONS)
    metrics = table[["MAE", "RMSE", "pinball", "coverage_80"]].to_numpy()
    assert np.isfinite(metrics).all()
    assert ((table["coverage_80"] >= 0) & (table["coverage_80"] <= 1)).all()
    assert (table["RMSE"] >= table["MAE"]).all()


def test_persistence_repeats_the_last_observed_value():
    wide = load.to_wide(load.load_run("run_01"))
    point, quant, origin = baselines.Persistence().fit([wide]).predict(wide)
    q = wide[load.TARGET].to_numpy()
    assert np.array_equal(point[:, 0], q[origin])
    assert (np.diff(quant, axis=-1) >= 0).all()   # quantiles do not cross
