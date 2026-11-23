"""Read the raw runs and the train/validation/test split.

You do not need this file to train a model: the tables in data/tables/ are
ready to use. make_tables.py uses it to build them.
"""
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

SAMPLE_MS = 2.0              # one sample every 2 ms
SAMPLES_PER_RUN = 45_000     # 90 s per run
BUFFER_PKTS = 64             # the queue holds at most 64 packets

TARGET = "qdepth_pkts_h0"    # queue length at hop 0 (the bottleneck switch)
FEATURES = ("qdepth_pkts_h0", "qdelay_us_h0", "qdepth_pkts_h1", "qdelay_us_h1")


def list_runs(data_dir=DATA_DIR):
    """Run ids found in data/runs, sorted."""
    return sorted(p.stem for p in (Path(data_dir) / "runs").glob("run_*.parquet"))


def load_run(run_id, data_dir=DATA_DIR):
    """One run as stored: one row per sample (seq) and switch (hop)."""
    return pd.read_parquet(Path(data_dir) / "runs" / f"{run_id}.parquet")


def load_splits(data_dir=DATA_DIR):
    """{'train': [...], 'validation': [...], 'test': [...]} run ids."""
    splits = json.loads((Path(data_dir) / "splits.json").read_text())
    return {k: splits[k] for k in ("train", "validation", "test")}


def to_wide(run):
    """One row per sample, from 0 to the last received one.

    Columns: t_ms, qdepth_pkts_h0, qdelay_us_h0, qdepth_pkts_h1, qdelay_us_h1
    (suffix _h0 = bottleneck switch, _h1 = downstream switch). Lost samples
    are rows of NaN, so row i is always sample i.
    """
    wide = run.pivot(index="seq", columns="hop", values=["qdepth_pkts", "qdelay_us"])
    wide.columns = [f"{name}_h{hop}" for name, hop in wide.columns]
    wide = wide[list(FEATURES)].astype("float64")
    wide.insert(0, "t_ms", run.groupby("seq")["t_ms"].first())
    return wide.reindex(pd.RangeIndex(0, int(run["seq"].max()) + 1, name="seq"))
