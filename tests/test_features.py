"""Tests de nowcast.ml_logic.features: target, leakage, rezagos y noticias."""

import numpy as np
import pandas as pd
import pytest

from nowcast import params
from nowcast.ml_logic.features import (build_features, build_model_table,
                                       get_feature_columns)

CHECK_DATES = ["2003-06-01", "2008-10-01", "2015-01-01", "2020-04-01", "2023-12-01"]
HIGH_FREQ_RAW = ["oil_mean", "oil_last", "gas_mean", "gas_last"]


def test_model_table_has_no_nan(table):
    assert not table[params.FEATURES + [params.TARGET]].isna().any().any()


def test_model_table_columns(table):
    assert set(params.FEATURES) <= set(table.columns)
    assert not set(params.DROP_FEATURES) & set(table.columns)
    assert (table[params.MONTH_COL] == table.index.month).all()
    assert get_feature_columns(use_news=False) == params.FEATURES


def test_target_is_log_change_of_cpi(raw):
    out = build_features(raw)
    t, prev = pd.Timestamp("2010-05-01"), pd.Timestamp("2010-04-01")
    expected = 100 * np.log(raw.loc[t, "cpi"] / raw.loc[prev, "cpi"])
    assert out.loc[t, "y"] == pytest.approx(expected)


@pytest.mark.parametrize("T", CHECK_DATES)
def test_features_do_not_change_when_future_data_changes(raw, T):
    """Features en T no dependen de nada posterior a T."""
    T = pd.Timestamp(T)
    future_changed = raw.copy()
    future_changed.loc[future_changed.index > T] *= 1.5

    base = build_features(raw).loc[T, params.ALL_FEATURES]
    pd.testing.assert_series_equal(
        build_features(future_changed).loc[T, params.ALL_FEATURES], base)
    pd.testing.assert_series_equal(
        build_features(raw.loc[:T]).loc[T, params.ALL_FEATURES], base)


@pytest.mark.parametrize("T", CHECK_DATES)
def test_availability_lags_hold_at_t(raw, T):
    """Series con rezago 1 (y el propio IPC): su valor en T no entra en las features de T."""
    T = pd.Timestamp(T)
    lagged_cols = [c for c in raw.columns if c not in HIGH_FREQ_RAW]
    changed = raw.copy()
    changed.loc[T, lagged_cols] *= 1.5

    base, new = build_features(raw).loc[T], build_features(changed).loc[T]
    pd.testing.assert_series_equal(new[params.ALL_FEATURES], base[params.ALL_FEATURES])
    assert new["y"] != base["y"]  # el target sí cambia


@pytest.mark.parametrize("col", HIGH_FREQ_RAW)
def test_same_month_series_do_enter_at_t(raw, col):
    """Control: oil/gas tienen rezago 0, así que cambiarlos en T SÍ mueve su feature."""
    T = pd.Timestamp("2015-01-01")
    changed = raw.copy()
    changed.loc[T, col] *= 1.5
    assert build_features(changed).loc[T, f"{col}_chg"] != \
        build_features(raw).loc[T, f"{col}_chg"]


def test_lag_and_moving_average_use_only_past_months(raw):
    out = build_features(raw)
    t = pd.Timestamp("2012-07-01")
    pos = out.index.get_loc(t)
    w = params.MA_WINDOW
    assert out.loc[t, f"y_ma{w}"] == pytest.approx(out["y"].iloc[pos - w:pos].mean())
    for k in range(1, params.N_CPI_LAGS + 1):
        assert out.loc[t, f"y_lag{k}"] == out["y"].iloc[pos - k]


def _news(start="2005-03-01", gap="2010-06-01"):
    idx = pd.date_range(start, params.FRED_END, freq="MS")
    news = pd.DataFrame({"infla_score_mean": np.linspace(-1, 1, len(idx)),
                         "infla_net_share": 0.2}, index=idx)
    return news.drop(pd.Timestamp(gap))


def test_news_ignored_when_use_news_false(raw, table):
    pd.testing.assert_frame_equal(
        build_model_table(raw, news=_news(), use_news=False), table)


def test_news_restricts_sample_when_use_news_true(raw):
    with_news = build_model_table(raw, news=_news(), use_news=True)
    assert with_news.index.min() == pd.Timestamp("2005-03-01")
    assert pd.Timestamp("2010-06-01") not in with_news.index
    assert not with_news[get_feature_columns(use_news=True)].isna().any().any()


def test_use_news_without_file_runs_macro_only(raw, table):
    pd.testing.assert_frame_equal(
        build_model_table(raw, news=None, use_news=True), table)
