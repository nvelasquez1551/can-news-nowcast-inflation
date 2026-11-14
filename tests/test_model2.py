"""Synthetic tests for the Model 2 evaluation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.model2.evaluate import compare_forecasts, diebold_mariano, mae, rmse


def test_rmse_and_mae_known_values():
    y = [1.0, 2.0, 3.0, 4.0]
    pred = [1.0, 1.0, 5.0, 4.0]
    assert rmse(y, pred) == pytest.approx(np.sqrt((0 + 1 + 4 + 0) / 4))
    assert mae(y, pred) == pytest.approx((0 + 1 + 2 + 0) / 4)


def test_dm_known_statistic_horizon_1():
    # d = e_a^2 - 0 = [1, 4, 9, 16]; mean 7.5; gamma0 = 129 / 4.
    result = diebold_mariano([1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0])
    dm = 7.5 / np.sqrt((129 / 4) / 4)
    hln = np.sqrt((4 + 1 - 2) / 4)
    assert result.statistic == pytest.approx(dm * hln)
    assert result.n == 4


def test_dm_known_statistic_horizon_2():
    e_a = np.array([1.0, -2.0, 3.0, -1.0, 2.0, 0.5])
    e_b = np.zeros(6)
    d = e_a**2
    c = d - d.mean()
    n = len(d)
    lrv = np.mean(c**2) + 2 * np.sum(c[1:] * c[:-1]) / n
    expected = d.mean() / np.sqrt(lrv / n) * np.sqrt((n + 1 - 4 + 2 / n) / n)
    result = diebold_mariano(e_a, e_b, horizon=2)
    assert result.statistic == pytest.approx(expected)


def test_dm_is_antisymmetric_and_two_sided_pvalue_matches():
    rng = np.random.default_rng(0)
    e_a = rng.normal(size=48)
    e_b = rng.normal(size=48)
    ab = diebold_mariano(e_a, e_b)
    ba = diebold_mariano(e_b, e_a)
    assert ab.statistic == pytest.approx(-ba.statistic)
    assert ab.pvalue == pytest.approx(ba.pvalue)
    assert 0.0 <= ab.pvalue <= 1.0


def test_dm_one_sided_detects_better_forecast():
    rng = np.random.default_rng(1)
    big = rng.normal(scale=2.0, size=200)
    small = rng.normal(scale=0.5, size=200)
    greater = diebold_mariano(big, small, alternative="greater")
    less = diebold_mariano(big, small, alternative="less")
    assert greater.statistic > 0
    assert greater.pvalue < 0.01
    assert greater.pvalue + less.pvalue == pytest.approx(1.0)


def test_dm_zero_variance_returns_nan():
    result = diebold_mariano([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
    assert np.isnan(result.statistic)
    assert np.isnan(result.pvalue)


def test_dm_rejects_nan_and_bad_inputs():
    with pytest.raises(ValueError):
        diebold_mariano([1.0, np.nan], [0.0, 0.0])
    with pytest.raises(ValueError):
        diebold_mariano([1.0, 2.0], [0.0])
    with pytest.raises(ValueError):
        diebold_mariano([1.0, 2.0], [0.0, 0.0], alternative="bigger")


def test_compare_forecasts_aligns_months_and_tests_against_baseline():
    frame = pd.DataFrame(
        {
            "y": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "ar": [0.0, 0.3, 0.1, 0.5, 0.3, 0.9],
            "xgb_macro": [0.1, 0.2, np.nan, 0.4, 0.5, 0.6],
            "xgb_news": [0.15, 0.25, 0.3, 0.35, 0.55, 0.65],
        }
    )
    table = compare_forecasts(
        frame, target="y", forecasts=["ar", "xgb_macro", "xgb_news"], baseline="ar"
    ).set_index("forecast")

    assert list(table.index) == ["ar", "xgb_macro", "xgb_news"]
    assert (table["n"] == 5).all()
    aligned = frame.dropna()
    assert table.loc["xgb_news", "rmse"] == pytest.approx(rmse(aligned["y"], aligned["xgb_news"]))
    assert table.loc["ar", "rmse_vs_baseline"] == pytest.approx(1.0)
    assert np.isnan(table.loc["ar", "dm_pvalue"])
    assert table.loc["xgb_news", "dm_stat"] > 0


def test_compare_forecasts_missing_column_raises():
    frame = pd.DataFrame({"y": [1.0], "ar": [1.0]})
    with pytest.raises(KeyError):
        compare_forecasts(frame, target="y", forecasts=["xgb"], baseline="ar")
