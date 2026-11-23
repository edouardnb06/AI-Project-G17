"""The score and the results file."""
import numpy as np
import pandas as pd
import pytest

from src import evaluate


def test_score_on_a_small_example():
    scores = evaluate.score([0, 2, 4, 6], [1, 2, 4, 3])     # errors 1, 0, 0, 3
    assert set(scores) == {"MAE", "RMSE"}
    assert scores["MAE"] == pytest.approx(1.0)
    assert scores["RMSE"] == pytest.approx(np.sqrt(10 / 4))


def test_perfect_predictions_score_zero():
    assert evaluate.score([3, 5], [3, 5]) == {"MAE": 0.0, "RMSE": 0.0}


def test_a_column_shaped_prediction_is_not_broadcast():
    # some models return shape (n, 1); (n,) minus (n, 1) would silently become (n, n)
    flat = evaluate.score([1, 2, 3], [1, 2, 5])
    column = evaluate.score([1, 2, 3], np.array([[1], [2], [5]]))
    assert column == flat


def test_score_refuses_different_lengths():
    with pytest.raises(ValueError):
        evaluate.score([1, 2, 3], [1, 2])


def test_add_result_creates_the_file_then_appends(tmp_path):
    path = tmp_path / "results" / "results.csv"
    evaluate.add_result("first", "val", {"MAE": 2.0, "RMSE": 3.0}, path=path)
    evaluate.add_result("second", "val", {"MAE": 1.0, "RMSE": 1.5}, path=path)
    table = pd.read_csv(path)
    assert list(table.columns) == ["name", "split", "MAE", "RMSE"]
    assert table["name"].tolist() == ["first", "second"]
    assert table["MAE"].tolist() == [2.0, 1.0]


def test_add_result_keeps_earlier_rows_when_the_last_line_has_no_newline(tmp_path):
    path = tmp_path / "results.csv"
    path.write_text("name,split,MAE,RMSE\nfirst,val,2.0,3.0")
    evaluate.add_result("second", "val", {"MAE": 1.0, "RMSE": 1.5}, path=path)
    assert pd.read_csv(path)["name"].tolist() == ["first", "second"]


def test_add_result_refuses_an_unknown_split(tmp_path):
    with pytest.raises(ValueError):
        evaluate.add_result("model", "validation", {"MAE": 1.0, "RMSE": 1.0}, path=tmp_path / "r.csv")


def test_shipped_results_file_has_the_right_columns():
    table = pd.read_csv(evaluate.RESULTS_CSV)
    assert list(table.columns) == ["name", "split", "MAE", "RMSE"]
    assert table["split"].isin(["train", "val", "test"]).all()
    assert (table["RMSE"] >= table["MAE"]).all()
