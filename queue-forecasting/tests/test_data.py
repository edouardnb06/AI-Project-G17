"""Row counts, columns and splits of the shipped dataset."""
import numpy as np
import pandas as pd

from src import load

N_RUNS = 39
TOTAL_ROWS = 3_486_250
COLUMNS = ["run_id", "seq", "t_ms", "hop", "qdepth_pkts", "qdelay_us"]


def test_run_count_and_total_rows():
    runs = load.list_runs()
    assert len(runs) == N_RUNS
    assert runs[0] == "run_01" and runs[-1] == f"run_{N_RUNS:02d}"
    assert sum(len(load.load_run(r)) for r in runs) == TOTAL_ROWS


def test_every_received_sample_has_both_hops():
    for run in load.list_runs():
        df = load.load_run(run)
        assert list(df.columns) == COLUMNS
        assert (df["run_id"] == run).all()
        per_seq = df.groupby("seq")["hop"].agg(["size", "min", "max"])
        assert (per_seq["size"] == 2).all() and (per_seq["min"] == 0).all() and (per_seq["max"] == 1).all()
        assert df["seq"].between(0, load.SAMPLES_PER_RUN - 1).all()
        assert df["qdepth_pkts"].between(0, load.BUFFER_PKTS).all()


def test_sample_csv_is_first_10_s_of_run_01():
    sample = pd.read_csv(load.DATA_DIR / "sample.csv")
    run = load.load_run("run_01")
    expected = run[run["t_ms"] < 10_000].reset_index(drop=True)
    assert list(sample.columns) == COLUMNS
    assert len(sample) == len(expected)
    assert np.array_equal(sample["seq"], expected["seq"])
    assert np.allclose(sample["t_ms"], expected["t_ms"])


def test_no_run_in_two_splits():
    splits = load.load_splits()
    names = ["train", "validation", "test"]
    sets = [set(splits[n]) for n in names]
    for i in range(3):
        for j in range(i + 1, 3):
            assert not sets[i] & sets[j], (names[i], names[j])
    assert set().union(*sets) == set(load.list_runs())
    assert [len(s) for s in sets] == [23, 8, 8]
