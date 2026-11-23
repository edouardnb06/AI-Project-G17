# Queue forecasting

**Task: Predict how many packets will be waiting in a switch's queue 20 ms from now, using its recent history.**

You need basic Python and scikit-learn. You do not need to know anything about networks.

## What is a queue?

Think of a supermarket checkout line. Customers join the line, and the cashier serves them one at a time. If customers come faster than the cashier can work, the line gets longer. If nobody comes for a while, the line becomes empty.

A network switch works the same way:

| supermarket | our data |
|---|---|
| a customer | a packet (a small piece of data) |
| the cashier | the output of the switch, which handles 750 packets per second |
| people waiting | the queue: packets waiting in the switch |
| the shop has room for 64 people in line | the queue holds at most 64 packets; when it is full, new packets are lost |

Every 2 ms, a small measurement packet goes through the switch and reports how many packets are waiting. That report is one **sample**. We have 39 recordings ("runs") of 90 seconds each.

Why predict it? A long queue means long waiting times and lost packets. If we know it 20 ms before, there is time to react.

## The tables

The data is ready to use. There is nothing to prepare.

| file | rows | runs | use it for |
|---|---:|---:|---|
| `data/tables/train.parquet` | 196,306 | 23 | teaching the model |
| `data/tables/val.parquet` | 68,097 | 8 | comparing models and settings |
| `data/tables/test.parquet` | 68,116 | 8 | **once, at the very end** (see `TASKS.md`) |

**One row is one moment of one run.** It answers two questions: what did the queue look like up to now (the features), and how long was it 20 ms later (`y`)?

| column | meaning |
|---|---|
| `run_id` | which run the row comes from (not a feature) |
| `t_ms` | time since the start of the run, in milliseconds (not a feature) |
| `y` | **what we predict:** the queue length 10 samples (20 ms) later |
| `lag_1` ... `lag_10` | the last 10 queue lengths. `lag_1` is the queue **now**, `lag_2` is one sample (2 ms) earlier, ..., `lag_10` is 9 samples (18 ms) earlier |
| `mean_10` | mean of the last 10 queue lengths |
| `mean_50` | mean of the last 50 queue lengths (the last 100 ms) |
| `max_50` | largest of the last 50 queue lengths |
| `slope_10` | how fast the queue is growing (+) or shrinking (-): packets per sample, from a straight line through the last 10 queue lengths |
| `delay_now_ms` | how long the newest measurement packet waited in the queue, in milliseconds |

Example, the first row of `train.parquet`: `lag_1 = 3` and `lag_2 = 4`, so 3 packets are waiting now and there were 4 a moment ago. `y = 1`, so 20 ms later only 1 was waiting.

All features use only the past and the present, never the future. Two rules to keep it that way:

- Never use `y`, `run_id` or `t_ms` as a feature.
- Do not mix the tables or split them again at random. Neighbouring rows of a run are almost the same, so a random split would let the model see the answers. The three tables come from different runs.

`data/DATA_CARD.md` has the details, including which moments are missing from the tables.

## Get started

Run these in the `queue-forecasting/` folder (Python 3.11 to 3.14):

```
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # macOS / Linux
pip install -r requirements.txt
jupyter lab notebooks/01_quickstart.ipynb
```

Start with `01_quickstart.ipynb`. It goes through every step once. Then choose a role in `TASKS.md`.

## Get a score

```python
from src.evaluate import score, add_result

scores = score(y_val, y_pred)            # {"MAE": ..., "RMSE": ...}
add_result("my model", "val", scores)    # adds one row to results/results.csv
```

Both numbers are in packets, and lower is better. `results/results.csv` is the comparison table of the group, and it fills itself: every call of `add_result` adds one row at the end. If you score the same model twice, it appears twice, and the last row counts.

The first number to beat is **persistence**: the guess that the queue stays as it is now.

## Folders

```
queue-forecasting/
├── README.md
├── TASKS.md               who does what, and what "done" means
├── requirements.txt       the packages to install
├── notebooks/             01 quickstart, then one starter notebook per model
├── data/
│   ├── tables/            train, val and test: start here
│   ├── DATA_CARD.md       where the data comes from, and its problems
│   ├── runs/              the raw recordings
│   ├── splits.json        which run belongs to train, validation or test
│   └── sample.csv         10 seconds of raw data, to look at
├── results/results.csv    the scores of all models
├── src/
│   ├── evaluate.py        score and add_result
│   ├── make_tables.py     builds the tables again (you do not need it)
│   └── load.py            reads the raw recordings
├── tests/                 run them with: pytest
└── extras/                optional, more advanced material
```

## Glossary

| word | meaning |
|---|---|
| **packet** | a small piece of data that travels through a network; a customer in the supermarket |
| **queue** | the packets waiting in a switch; the checkout line |
| **sample** | one measurement of the queue; there is one every 2 ms |
| **horizon** | how far ahead we predict: 10 samples, which is 20 ms |
| **feature** | a column the model may use to make its prediction |
| **MAE** | mean absolute error: how many packets the prediction is off, on average |
| **RMSE** | root mean squared error: like MAE, but large mistakes count more |

## Credit

Data collected by Mohamed Ali Kabiri on his emulated network testbed (BMv2 software switches in Mininet).
