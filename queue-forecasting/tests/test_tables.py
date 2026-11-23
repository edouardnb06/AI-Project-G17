"""The ready-made tables: correct values, no peeking at the future, no gaps."""
import numpy as np
import pandas as pd
import pytest

from src import load, make_tables

COLUMNS = (["run_id", "t_ms", "y"] + [f"lag_{k}" for k in range(1, 11)]
           + ["mean_10", "mean_50", "max_50", "slope_10", "delay_now_ms"])
FEATURES = COLUMNS[3:]
SPLITS = {"train": "train", "val": "validation", "test": "test"}   # table name -> name in splits.json
MAX_BYTES = 50 * 1024 * 1024


def toy_run(n=200, lost=()):
    """A run where the queue length at sample i is i % 50; `lost` samples are missing."""
    rows = []
    for seq in range(n):
        if seq in lost:
            continue
        rows.append(("run_x", seq, 2.0 * seq, 0, seq % 50, 1000 * (seq % 50)))
        rows.append(("run_x", seq, 2.0 * seq, 1, 0, 50))
    return pd.DataFrame(rows, columns=["run_id", "seq", "t_ms", "hop", "qdepth_pkts", "qdelay_us"])


def samples(table):
    """Sample numbers of the rows of a toy table (toy samples are exactly 2 ms apart)."""
    return (table["t_ms"] / 2).astype(int).to_list()


def test_columns_and_values_of_one_row():
    table = make_tables.make_table(toy_run(), every=1)
    assert list(table.columns) == COLUMNS
    row = table[table["t_ms"] == 2.0 * 120].iloc[0]        # sample 120, queue length 20
    assert row["run_id"] == "run_x"
    assert row["y"] == 130 % 50                            # 10 samples later
    assert [row[f"lag_{k}"] for k in range(1, 11)] == list(range(20, 10, -1))   # lag_1 is now
    assert row["mean_10"] == pytest.approx(15.5)           # mean of 11 ... 20
    assert row["mean_50"] == pytest.approx(24.5)           # one full cycle 0 ... 49
    assert row["max_50"] == 49
    assert row["slope_10"] == pytest.approx(1.0)           # one packet more per sample
    assert row["delay_now_ms"] == pytest.approx(20.0)      # 20,000 microseconds


def test_slope_is_the_fitted_line_through_the_last_10():
    table = make_tables.make_table(load.load_run("run_01"), every=50)
    oldest_first = table[[f"lag_{k}" for k in range(10, 0, -1)]].to_numpy(dtype=float)
    expected = np.polyfit(np.arange(10), oldest_first.T, deg=1)[0]
    assert np.allclose(table["slope_10"], expected)


def test_rows_stay_inside_the_run():
    # a row needs 50 past samples and one 10 samples later: samples 49 ... 189 of 0 ... 199
    table = make_tables.make_table(toy_run(n=200), every=1)
    assert samples(table) == list(range(49, 190))


def test_rows_do_not_cross_a_gap():
    # sample 100 is lost, so no row may use it: not as history (100 ... 149),
    # not as target (90), and not in between (91 ... 99)
    table = make_tables.make_table(toy_run(n=200, lost=[100]), every=1)
    assert samples(table) == list(range(49, 90)) + list(range(150, 190))


def test_keeps_every_fifth_row():
    everything = make_tables.make_table(toy_run(), every=1)
    fifth = make_tables.make_table(toy_run(), every=5)
    pd.testing.assert_frame_equal(fifth, everything.iloc[::5].reset_index(drop=True))


def test_features_do_not_use_the_future():
    run = load.load_run("run_01")
    changed = run.copy()
    later = changed["seq"] > 20_000
    changed.loc[later, "qdepth_pkts"] = 64 - changed.loc[later, "qdepth_pkts"]
    changed.loc[later, "qdelay_us"] += 5_000
    limit = run.loc[run["seq"] <= 20_000, "t_ms"].max()
    before = make_tables.make_table(run, every=1)
    after = make_tables.make_table(changed, every=1)
    past = before["t_ms"] <= limit
    assert past.sum() > 10_000
    pd.testing.assert_frame_equal(before.loc[past, FEATURES], after.loc[past, FEATURES])
    assert (before.loc[past, "y"] != after.loc[past, "y"]).any()   # only the target looks ahead


@pytest.mark.parametrize("name", SPLITS)
def test_shipped_table(name):
    path = make_tables.TABLES_DIR / f"{name}.parquet"
    assert path.stat().st_size <= MAX_BYTES
    table = pd.read_parquet(path)
    assert list(table.columns) == COLUMNS
    assert table["run_id"].unique().tolist() == load.load_splits()[SPLITS[name]]
    assert not table.isna().any().any()
    queue = table[["y", "max_50"] + [f"lag_{k}" for k in range(1, 11)]]
    assert queue.to_numpy().min() >= 0 and queue.to_numpy().max() <= load.BUFFER_PKTS
    assert (table["delay_now_ms"] >= 0).all()


@pytest.mark.parametrize("name", SPLITS)
def test_shipped_table_is_what_the_script_builds(name):
    table = pd.read_parquet(make_tables.TABLES_DIR / f"{name}.parquet")
    run_id = table["run_id"].iloc[-1]
    shipped = table[table["run_id"] == run_id].reset_index(drop=True)
    pd.testing.assert_frame_equal(shipped, make_tables.make_table(load.load_run(run_id)))
