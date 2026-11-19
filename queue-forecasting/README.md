# Queue forecasting

**Topic: Multi-horizon forecasting of switch queue occupancy from high-frequency telemetry, with prediction intervals.**

## Task

Every 2 ms, a measurement packet reports the queue depth and queue delay at two switches in series. Given this telemetry up to sample *t*, forecast the depth of the bottleneck queue (`qdepth_pkts` at hop 0) at four horizons:

| horizon | samples ahead | time ahead |
|---|---:|---:|
| t+1 | 1 | 2 ms |
| t+5 | 5 | 10 ms |
| t+10 | 10 | 20 ms |
| t+25 | 25 | 50 ms |

For each horizon, a model gives a point forecast and the 0.1, 0.5 and 0.9 quantiles. The interval from the 0.1 to the 0.9 quantile is the 80% prediction interval.

## Metrics

`src/evaluate.py` computes all of these per horizon:

- **MAE** and **RMSE** of the point forecast
- **pinball loss** for the quantiles 0.1, 0.5 and 0.9, and their mean
- **80% interval coverage**: the share of targets that fall between the 0.1 and 0.9 quantiles, which should be close to 0.80

Targets that were lost, or that lie past the end of a run, are left out.

## Data

The dataset has 39 runs of 90 s each, one Parquet file per run in `data/runs/`. Each file has the columns `run_id`, `seq`, `t_ms`, `hop`, `qdepth_pkts` and `qdelay_us`. See [`data/DATA_CARD.md`](data/DATA_CARD.md) for units, setup, known quirks, row counts and sizes. `data/sample.csv` holds the first 10 s of `run_01`.

The split is by run and fixed in `data/splits.json`: 23 train, 8 validation and 8 test runs.

- Fit models on train.
- Choose between them on validation, or with cross-validation grouped by run on the training runs.
- Use the test runs **once, at the end**.

## Models

The models are up to the group: statistical, gradient boosting, recurrent, and attention or TCN. The baselines to beat are in `src/baselines.py`:

- **persistence**: the depth at t+h is the depth at t
- **EWMA**: an exponentially weighted mean, with alpha chosen per horizon on the training runs

Both get their quantiles by adding the training-residual quantiles to the point forecast.

## Layout

```
queue-forecasting/
├── data/
│   ├── DATA_CARD.md
│   ├── runs/run_01.parquet ... run_39.parquet
│   ├── sample.csv
│   └── splits.json
├── notebooks/00_explore.ipynb    first look at the data
├── src/
│   ├── load.py                   runs, splits, wide format, supervised windows
│   ├── evaluate.py               metrics per horizon
│   └── baselines.py              persistence and EWMA
├── tests/                        data checks, metrics, baselines end to end
├── pytest.ini
└── requirements.txt              pinned, Python 3.11 to 3.14
```

## Setup and commands

Run every command from the `queue-forecasting/` folder.

```
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # macOS / Linux
pip install -r requirements.txt

pytest                                  # data checks, metrics, baselines
python -m src.baselines                 # baseline table on the validation runs
python -m src.baselines --split test    # only once, at the end
jupyter lab notebooks/00_explore.ipynb
```

## Using the loaders

```python
from src import load, evaluate

splits = load.load_splits()                        # {'train': [...], 'validation': [...], 'test': [...]}
wide = load.to_wide(load.load_run("run_01"))      # one row per sample, NaN rows for lost samples
X, Y, origin = load.make_windows(wide, lookback=50)
# X: (n, 50, 4) last 50 samples of both depths and delays; Y: (n, 4) depth at t+1, 5, 10, 25

X, Y, meta = load.load_windows("train", lookback=50, stride=1)   # meta: run_id, seq of each row
table = evaluate.evaluate(Y_true, point, quantiles, load.HORIZONS)  # point (n, 4), quantiles (n, 4, 3)
```

With `lookback=50`, all four features and `stride=1`, the training windows take about 0.8 GB as float32, and about twice that while they are being built. On a small machine, use a larger `stride` or a shorter `lookback`.

## Credit

Data collected by Mohamed Ali Kabiri on his emulated network testbed (BMv2 software switches in Mininet).
