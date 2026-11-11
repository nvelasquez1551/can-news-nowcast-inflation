"""Tests de nowcast.ml_logic.model: pipelines, métricas, Diebold-Mariano y expanding."""

from functools import partial

import numpy as np
import pandas as pd
import pytest

from nowcast import params
from nowcast.ml_logic import model
from nowcast.ml_logic.data import split_by_date
from nowcast.ml_logic.model import (build_model, compute_metrics, cross_validate,
                                    diebold_mariano, evaluate, expanding_forecast,
                                    search_summary)
from tests.conftest import SMALL_GRID


@pytest.fixture
def small_grid(monkeypatch):
    """expanding_forecast llama a grid_search: usar una grilla mínima."""
    monkeypatch.setattr(model, "grid_search",
                        partial(model.grid_search, grid=SMALL_GRID))


def test_linear_pipeline_recovers_exact_linear_relation(table):
    beta = np.arange(1, len(params.NUMERIC_FEATURES) + 1) / 10
    table = table.assign(y=table[params.NUMERIC_FEATURES].to_numpy() @ beta + 0.5)
    train, test = split_by_date(table)

    pipe = build_model("linear").fit(train[params.FEATURES], train["y"])
    np.testing.assert_allclose(pipe.predict(test[params.FEATURES]), test["y"], atol=1e-8)
    assert cross_validate(build_model("linear"), train[params.FEATURES],
                          train["y"]).max() < 1e-8


def test_cross_validate_returns_one_rmse_per_fold(table):
    train, _ = split_by_date(table)
    folds = cross_validate(build_model("linear"), train[params.FEATURES], train["y"])
    assert folds.shape == (params.N_SPLITS,) and (folds > 0).all()


def test_grid_search_summary_is_consistent(table):
    train, _ = split_by_date(table)
    summary = search_summary(model.grid_search(train[params.FEATURES], train["y"],
                                               grid=SMALL_GRID))
    assert summary["best_params"]["model__n_estimators"] in [5, 10]
    assert summary["cv_rmse_folds"].shape == (params.N_SPLITS,)
    assert summary["cv_rmse_mean"] == pytest.approx(summary["cv_rmse_folds"].mean())
    assert summary["cv_rmse_std"] == pytest.approx(summary["cv_rmse_folds"].std())


def test_build_model_applies_params():
    pipe = build_model("xgb", model_params={"model__max_depth": 2})
    assert pipe.named_steps["model"].get_params()["max_depth"] == 2
    assert pipe.named_steps["model"].get_params()["random_state"] == params.SEED
    with pytest.raises(ValueError):
        build_model("rf")


def test_compute_metrics_by_hand():
    idx = pd.date_range(params.TEST_START, params.TEST_END, freq="MS")
    errors = np.where(idx.year <= 2021, 1.0, 3.0)  # error 1 en 2020-21, 3 en 2022-23
    preds = pd.DataFrame({"y": 0.0, "m": errors}, index=idx)
    row = compute_metrics(preds, ["m"]).iloc[0]
    assert row["rmse_2020_2021"] == pytest.approx(1.0)
    assert row["rmse_2022_2023"] == pytest.approx(3.0)
    assert row["mae_full"] == pytest.approx(2.0)
    assert row["rmse_full"] == pytest.approx(np.sqrt(5.0))


def test_diebold_mariano_by_hand():
    rng = np.random.default_rng(1)
    y = rng.normal(size=48)
    preds = pd.DataFrame({"y": y,
                          "a": y + 0.1 * rng.normal(size=48),  # A mucho mejor
                          "b": y + 1.0 * rng.normal(size=48)})
    table = diebold_mariano(preds, pairs=[("a", "b")])

    d = (preds["y"] - preds["a"]) ** 2 - (preds["y"] - preds["b"]) ** 2
    n = len(d)
    expected = d.mean() / np.sqrt(d.var(ddof=0) / n) * np.sqrt((n - 1) / n)
    assert table.loc[0, "dm_stat"] == pytest.approx(expected)
    assert table.loc[0, "dm_stat"] < 0 and table.loc[0, "p_value"] < 0.01


def test_evaluate_baselines(table):
    train, test = split_by_date(table)
    pipe = build_model("linear").fit(train[params.FEATURES], train["y"])
    preds, results = evaluate({"linear": pipe}, train, test)

    assert (preds["naive_lag1"] == test["y_lag1"]).all()
    assert (preds["train_mean"] == train["y"].mean()).all()  # media FIJA de train
    assert list(results["model"]) == params.BASELINES + ["linear"]


def test_expanding_trains_only_on_past_months(table):
    out = expanding_forecast("linear", table)
    train, test = split_by_date(table)

    assert list(out["month"]) == list(test.index)
    assert (out["last_train_month"] < out["month"]).all()
    assert out["n_train"].iloc[0] == len(train)
    assert (out["n_train"].diff().dropna() == 1).all()
    for _, row in out.iloc[[0, 17, 47]].iterrows():
        history = table.loc[:row["month"]].iloc[:-1]
        assert row["train_mean"] == pytest.approx(history["y"].mean())
        assert row["naive_lag1"] == table.loc[row["month"], "y_lag1"]


def test_expanding_forecast_ignores_target_from_t_onwards(table):
    """Cambiar y desde t en adelante no puede cambiar la predicción de t ni las anteriores."""
    t = pd.Timestamp("2021-06-01")
    changed = table.copy()
    changed.loc[t:, "y"] += 10.0

    base = expanding_forecast("linear", table).set_index("month")["prediction"]
    new = expanding_forecast("linear", changed).set_index("month")["prediction"]
    pd.testing.assert_series_equal(new.loc[:t], base.loc[:t])
    assert not np.allclose(new.loc[t + pd.DateOffset(months=1):], base.loc[t + pd.DateOffset(months=1):])


def test_expanding_xgb_retunes_every_january(table, small_grid):
    out = expanding_forecast("xgb", table, retune_every=12)
    retuned = out.loc[out["retuned"], "month"]
    assert list(retuned.dt.strftime("%Y-%m")) == ["2020-01", "2021-01", "2022-01", "2023-01"]
    assert all(set(h) == set(SMALL_GRID) for h in out["hyperparameters"])


def test_expanding_xgb_fixed_params(table, small_grid):
    fixed = {"model__n_estimators": 5}
    out = expanding_forecast("xgb", table, retune_every=None, fixed_params=fixed)
    assert not out["retuned"].any()
    assert all(h == fixed for h in out["hyperparameters"])
    with pytest.raises(ValueError):
        expanding_forecast("xgb", table, retune_every=None)
