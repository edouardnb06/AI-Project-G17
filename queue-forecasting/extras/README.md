# Extras (optional)

This folder holds the more advanced version of the task. **You do not need it for the project.** Nothing in `notebooks/`, `TASKS.md` or `results/` depends on it.

Two things are harder here than in the main task:

- **Several horizons:** predict the queue 1, 5, 10 and 25 samples ahead (2, 10, 20 and 50 ms) instead of only 10.
- **Prediction intervals:** give a range as well as one number, as the 0.1, 0.5 and 0.9 quantiles. They are checked with the pinball loss and with the coverage of the 80% interval (the share of true values inside the range, which should be close to 0.80).

| file | content |
|---|---|
| `00_explore.ipynb` | a closer look at the raw training runs: both switches, lost samples, how quickly the queue forgets its past |
| `windows.py` | cuts the raw runs into inputs and targets for several horizons |
| `evaluate.py` | MAE, RMSE, pinball loss and interval coverage per horizon |
| `baselines.py` | persistence and EWMA (a weighted mean where recent samples count more), with intervals |
| `tests/` | tests for these files; `pytest` runs them too |

Run from the `queue-forecasting/` folder:

```
python -m extras.baselines        # baseline table on the validation runs
jupyter lab extras/00_explore.ipynb
```

The numbers here are not comparable with `results/results.csv`. The code in this folder uses the raw runs and also covers moments next to lost samples, which the tables leave out. The rule about the test runs is the same: once, at the end.
