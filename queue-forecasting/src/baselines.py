"""Persistence and EWMA baselines, with intervals from training residuals.

Both baselines forecast the bottleneck queue depth at t + h from the samples up
to t. Quantile forecasts are the point forecast plus the 0.1/0.5/0.9 quantiles
of the training-run residuals for that horizon, clipped to [0, buffer size].

Run from the queue-forecasting folder:
    python -m src.baselines                 # fit on train, report validation
    python -m src.baselines --split test    # once, at the very end
"""
import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # also allow `python src/baselines.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import evaluate, load


def origins_and_targets(wide, horizons=load.HORIZONS):
    """Origins = received bottleneck samples; Y[:, i] = depth at origin + horizons[i]."""
    q = wide[load.TARGET].to_numpy(dtype=float)
    origin = np.flatnonzero(~np.isnan(q))
    padded = np.concatenate([q, np.full(max(horizons), np.nan)])
    return origin, np.stack([padded[origin + h] for h in horizons], axis=1)


def ewma(q, alpha):
    """Exponentially weighted mean of the received samples; lost samples hold it."""
    return pd.Series(q).ewm(alpha=alpha, adjust=False, ignore_na=True).mean().to_numpy()


class Baseline:
    """Point forecast from level(); residual quantiles fitted on training runs."""

    def __init__(self, horizons=load.HORIZONS):
        self.horizons = tuple(horizons)

    def level(self, q, i):
        """Point forecast for horizon i at every sample, from the depth series q."""
        raise NotImplementedError

    def point(self, wide):
        q = wide[load.TARGET].to_numpy(dtype=float)
        origin, _ = origins_and_targets(wide, self.horizons)
        point = np.stack([self.level(q, i)[origin] for i in range(len(self.horizons))], axis=1)
        return point, origin

    def fit(self, train):
        residuals = []
        for wide in train:
            point, _ = self.point(wide)
            _, Y = origins_and_targets(wide, self.horizons)
            residuals.append(Y - point)
        residuals = np.concatenate(residuals)
        self.offsets = np.stack(  # (H, 3)
            [np.nanquantile(residuals[:, i], evaluate.QUANTILES) for i in range(len(self.horizons))])
        return self

    def predict(self, wide):
        """point (n, H), quantiles (n, H, 3) and the origin seq of each row."""
        point, origin = self.point(wide)
        quant = np.clip(point[:, :, None] + self.offsets[None], 0, load.BUFFER_PKTS)
        return point, quant, origin


class Persistence(Baseline):
    """The depth at t + h is the depth at t."""

    def level(self, q, i):
        return q


class EWMA(Baseline):
    """The depth at t + h is the EWMA at t; alpha per horizon chosen on train by MAE."""

    GRID = (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8)

    def fit(self, train):
        abs_err = np.zeros((len(self.GRID), len(self.horizons)))
        count = np.zeros_like(abs_err)
        for wide in train:
            q = wide[load.TARGET].to_numpy(dtype=float)
            origin, Y = origins_and_targets(wide, self.horizons)
            for j, alpha in enumerate(self.GRID):
                err = np.abs(Y - ewma(q, alpha)[origin, None])
                abs_err[j] += np.nansum(err, axis=0)
                count[j] += np.isfinite(err).sum(axis=0)
        self.alpha = np.array(self.GRID)[np.argmin(abs_err / count, axis=0)]
        return super().fit(train)

    def level(self, q, i):
        return ewma(q, self.alpha[i])


def run(train_ids, eval_ids, horizons=load.HORIZONS):
    """Fit both baselines on train_ids; one table row per model and horizon on eval_ids."""
    train = list(load.load_wide(train_ids).values())
    evals = list(load.load_wide(eval_ids).values())
    Y = np.concatenate([origins_and_targets(w, horizons)[1] for w in evals])
    tables = []
    for name, model in [("persistence", Persistence(horizons)), ("ewma", EWMA(horizons))]:
        model.fit(train)
        preds = [model.predict(w) for w in evals]
        point = np.concatenate([p[0] for p in preds])
        quant = np.concatenate([p[1] for p in preds])
        table = evaluate.evaluate(Y, point, quant, horizons, load.SAMPLE_MS).reset_index()
        table.insert(0, "model", name)
        table["alpha"] = getattr(model, "alpha", np.nan)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Persistence and EWMA baselines.")
    parser.add_argument("--split", choices=["validation", "test"], default="validation",
                        help="runs to report on (use test only once, at the end)")
    args = parser.parse_args()
    splits = load.load_splits()
    table = run(splits["train"], splits[args.split])
    print(f"Fitted on {len(splits['train'])} train runs, "
          f"evaluated on {len(splits[args.split])} {args.split} runs "
          f"(target: bottleneck queue depth, packets)\n")
    cols = ["model", "horizon", "ms", "n", "MAE", "RMSE", "pinball_0.1", "pinball_0.5",
            "pinball_0.9", "pinball", "coverage_80", "alpha"]
    print(table[cols].to_string(index=False, float_format="{:.3f}".format, na_rep="-"))


if __name__ == "__main__":
    main()
