# Tasks

Write your name next to a role. One person can take more than one role.

**The rule for everyone: `test.parquet` is used once, at the end, by the comparison role.** Everybody else works with `train.parquet` and `val.parquet` only. If we looked at test earlier, we would choose the settings that happen to fit it, and the final scores would look better than they are.

| role | notebook | who |
|---|---|---|
| A. Data, quickstart and baselines | `01_quickstart.ipynb`, `02_baselines.ipynb` | Mohamed Ali |
| B. Ridge regression | `03_linear_ridge.ipynb` | |Edouard de Noblet |
| C. Random forest | `04_random_forest.ipynb` | |
| D. Gradient boosting | `05_gradient_boosting.ipynb` | |
| E. Neural network | `06_neural_net.ipynb` | |
| F. Final comparison and report section | new: `07_comparison.ipynb` | |

## Before you start (everyone)

- [ ] I installed the packages (`README.md`, "Get started").
- [ ] I ran `01_quickstart.ipynb` from top to bottom.
- [ ] I can say what one row of the table means.

## A. Data, quickstart and baselines (Mohamed Ali)

Done means:

- [ ] The three tables are in `data/tables/`, and `pytest` passes.
- [ ] `data/DATA_CARD.md` explains the columns, the dropped rows and the known problems.
- [ ] `01_quickstart.ipynb` and `02_baselines.ipynb` run from top to bottom without errors.
- [ ] `persistence`, `mean_10` and `linear_regression` are in `results/results.csv` (split `val`).
- [ ] I answered the group's questions about the data.
- [ ] I wrote the data part of the report.

## B, C, D, E. One model each

Done means, for your notebook:

- [ ] It runs from top to bottom without errors.
- [ ] I can explain in 3 sentences how my model works and what its main settings do.
- [ ] The default version is in `results/results.csv` (split `val`).
- [ ] I tried at least 5 versions with other settings, trained on `train` and compared on `val`.
- [ ] My best version is in `results/results.csv` (split `val`) with a clear name, for example `ridge alpha=10`.
- [ ] I compared it with `persistence` and wrote down whether it is better, and by how much.
- [ ] I looked at the predicted vs actual plot and described one kind of moment where the model is wrong.
- [ ] "Your notes" is filled in, including what did not help.
- [ ] The final settings are in one model cell, so that role F can train the model again.
- [ ] I did not open `test.parquet`.

What to look at for each model (the notebook has the full list):

- **B. Ridge:** `alpha`, scaling, fewer or new features.
- **C. Random forest:** `min_samples_leaf`, `max_depth`, `max_features`, and which features matter.
- **D. Gradient boosting:** `learning_rate`, `max_iter`, `max_leaf_nodes`.
- **E. Neural network:** `early_stopping`, `hidden_layer_sizes`, `alpha`. An LSTM is optional.

## F. Final comparison and report section

Start when B, C, D and E are done. Done means:

- [ ] Everyone has confirmed that their model is final. After this point, no settings change.
- [ ] One plot compares all models on `val`, from `results/results.csv` (if a name appears twice, use the last row), with `persistence` as the reference.
- [ ] **Once:** each final model, the two baselines and the linear regression were trained on `train` and scored on `test.parquet`. The scores were added with `add_result(name, "test", scores)`.
- [ ] One plot and one table show the final test scores (MAE and RMSE) of all models.
- [ ] Nobody changed a model after seeing the test scores.
- [ ] The report section says which model is best, by how much it beats `persistence`, how the val and test scores differ, and what the limits are (see "Rows that are not in the tables" in `data/DATA_CARD.md`).

## Working together

- Only edit the notebook of your role. If you need a change in `src/`, ask the group first.
- `results/results.csv` is shared. If git reports a conflict there, keep the rows of both sides.
- A new package must be agreed with the group, because everyone has to install it.
