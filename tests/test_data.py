"""Tests de nowcast.ml_logic.data: carga, índice de noticias y split por fecha."""

import pandas as pd
import pytest

from nowcast import params
from nowcast.ml_logic.data import (download_fred, load_news_index, load_raw_data,
                                   split_by_date)


def test_split_does_not_overlap(table):
    train, test = split_by_date(table)
    assert train.index.max() < test.index.min()
    assert train.index.intersection(test.index).empty


def test_split_boundaries_and_sizes(table):
    train, test = split_by_date(table)
    assert train.index.max() == pd.Timestamp(params.TRAIN_END).replace(day=1)
    assert test.index.min() == pd.Timestamp(params.TEST_START)
    assert test.index.max() == pd.Timestamp(params.TEST_END).replace(day=1)
    assert len(test) == params.N_TEST_MONTHS
    # No se pierde ni se duplica ninguna fila dentro del rango
    in_range = table.loc[params.TRAIN_START:params.TEST_END]
    assert len(train) + len(test) == len(in_range)
    assert set(train["split"]) == {"train"} and set(test["split"]) == {"test"}


def test_split_fails_if_test_is_incomplete(table):
    with pytest.raises(AssertionError, match="test months"):
        split_by_date(table.iloc[:-1])


def test_load_raw_data_roundtrip(raw, tmp_path):
    path = tmp_path / "raw.csv"
    raw.to_csv(path)
    loaded = load_raw_data(path)
    pd.testing.assert_frame_equal(loaded, raw, check_freq=False)
    assert loaded.index.freqstr == "MS"


def test_load_raw_data_rejects_missing_columns(raw, tmp_path):
    path = tmp_path / "raw.csv"
    raw.drop(columns="gas_mean").to_csv(path)
    with pytest.raises(AssertionError, match="gas_mean"):
        load_raw_data(path)


def test_load_news_index_missing_file_returns_none(tmp_path):
    assert load_news_index(tmp_path / "does_not_exist.csv") is None


def test_load_news_index_aligns_to_month_start(tmp_path):
    path = tmp_path / "news.csv"
    pd.DataFrame({
        "month": ["2010-01-15", "2010-02-01"],  # cualquier día -> inicio de mes
        "infla_score_mean": [0.1, 0.2],
        "infla_net_share": [0.3, 0.4],
        "news_n_articles": [50, 60],  # columna extra: se ignora
    }).to_csv(path, index=False)

    news = load_news_index(path)
    assert list(news.columns) == params.NEWS_COLS
    assert list(news.index) == [pd.Timestamp("2010-01-01"), pd.Timestamp("2010-02-01")]


def test_load_news_index_rejects_missing_columns(tmp_path):
    path = tmp_path / "news.csv"
    pd.DataFrame({"month": ["2010-01-01"], "infla_score_mean": [0.1]}).to_csv(
        path, index=False)
    with pytest.raises(AssertionError, match="infla_net_share"):
        load_news_index(path)


def test_download_fred_never_overwrites_existing_raw(tmp_path):
    path = tmp_path / "raw.csv"
    path.write_text("keep me")
    with pytest.raises(FileExistsError):
        download_fred(path)  # falla antes de llamar a la API
    assert path.read_text() == "keep me"
