"""Score predictions and collect the scores of all models in one table."""
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_CSV = Path(__file__).resolve().parents[1] / "results" / "results.csv"
SPLITS = ("train", "val", "test")


def score(y_true, y_pred):
    """MAE and RMSE in packets, e.g. {"MAE": 1.2, "RMSE": 2.3}. Lower is better."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if len(y_true) != len(y_pred):
        raise ValueError(f"{len(y_true)} true values but {len(y_pred)} predictions")
    errors = y_pred - y_true
    return {"MAE": float(np.mean(np.abs(errors))),
            "RMSE": float(np.sqrt(np.mean(errors ** 2)))}


def add_result(name, split, scores, path=RESULTS_CSV):
    """Add one row (name, split, MAE, RMSE) at the end of results/results.csv.

    name    your model, e.g. "ridge alpha=10"
    split   "train", "val" or "test"
    scores  what score() returned
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
    path = Path(path)
    row = pd.DataFrame([{"name": name, "split": split,
                         "MAE": round(scores["MAE"], 4), "RMSE": round(scores["RMSE"], 4)}])
    if path.exists():
        row = pd.concat([pd.read_csv(path), row], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(path, index=False, lineterminator="\n")
