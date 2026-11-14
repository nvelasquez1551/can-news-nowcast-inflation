"""Tests de la API con datos y modelo sintéticos (no lee data/ ni models/)."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from nowcast import params
from nowcast.api import fast
from nowcast.ml_logic.data import split_by_date
from nowcast.ml_logic.model import build_model


def news_table(table: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    news = pd.DataFrame(rng.uniform(-1, 1, (len(table), len(params.NEWS_COLS))),
                        index=table.index, columns=params.NEWS_COLS)
    return table.join(news)


@pytest.fixture
def client(table, monkeypatch, tmp_path):
    tables = {"macro": table, "news": news_table(table)}
    fitted = {}
    for name, cols, key in (("linear", params.FEATURES, "macro"),
                            ("linear_news", params.FEATURES + params.NEWS_COLS, "news")):
        train, _ = split_by_date(tables[key])
        pipe = build_model("linear", cols).fit(train[cols], train[params.TARGET])
        fitted[name] = (pipe, {"model_name": name, "trained_at": "2026-01-01T00:00:00",
                               "train_start": "1999-02", "train_end": "2019-12",
                               "features": cols})

    def get_model(model_name):
        if model_name not in fitted:
            raise fast.HTTPException(404, "unknown model")
        return fitted[model_name]

    def get_table(features):
        return tables["news" if params.NEWS_COLS[0] in features else "macro"]

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "linear_20260101-000000-000000.joblib").touch()
    monkeypatch.setattr(params, "MODELS_DIR", models_dir)
    monkeypatch.setattr(fast, "get_model", get_model)
    monkeypatch.setattr(fast, "get_table", get_table)
    monkeypatch.setattr(params, "NEWS_SCORED_PATH", tmp_path / "news_scored.parquet")
    caches = (fast.load_scored_news, fast.keyword_counts, fast.same_sample_baseline)
    for cached in caches:
        cached.cache_clear()
    yield TestClient(fast.app)
    for cached in caches:
        cached.cache_clear()


def add_news_model():
    (params.MODELS_DIR / "linear_news_20260101-000000-000000.joblib").touch()


def test_root_and_months(client):
    root = client.get("/").json()
    assert root["status"] == "ok" and root["uses_news"] is False
    months = client.get("/months").json()
    assert months["latest"] == "2023-12"
    assert months["months"][0] > root["first_month"]  # el primer mes no tiene mes anterior


def test_nowcast_explains_the_change(client):
    body = client.get("/nowcast", params={"month": "2021-06"}).json()
    assert body["month"] == "2021-06" and body["previous_month"] == "2021-05"
    assert body["in_sample"] is False
    total = sum(driver["change"] for driver in body["drivers"])
    assert total == pytest.approx(body["change"], abs=1e-9)
    assert body["summary"].startswith("El nowcast de 2021-06")
    assert body["nowcast_annualized"] == pytest.approx(
        100 * ((1 + body["nowcast"] / 100) ** 12 - 1))


def test_nowcast_defaults_to_latest_and_validates_month(client):
    assert client.get("/nowcast").json()["month"] == "2023-12"
    assert client.get("/nowcast", params={"month": "abc"}).status_code == 422
    assert client.get("/nowcast", params={"month": "1990-01"}).status_code == 404
    assert client.get("/nowcast", params={"model": "rf"}).status_code == 404


def test_history_matches_table(client, table):
    rows = client.get("/history").json()["rows"]
    assert len(rows) == len(table)
    assert rows[-1]["month"] == "2023-12" and rows[-1]["in_sample"] is False
    assert rows[0]["in_sample"] is True


def test_headlines_without_and_with_scored_news(client):
    assert client.get("/headlines", params={"month": "2021-06"}).json()["available"] is False

    pd.DataFrame({
        "Date": pd.to_datetime(["2021-06-02", "2021-06-10", "2021-06-20", "2021-07-01"], utc=True),
        "Article_title": ["prices soar", "prices ease", "prices flat", "july news"],
        "infla_score": [0.9, -0.7, 0.0, 0.5],
        "infla_role": ["up", "down", "neu", "up"],
    }).to_parquet(params.NEWS_SCORED_PATH)
    fast.load_scored_news.cache_clear()

    body = client.get("/headlines", params={"month": "2021-06", "k": 1}).json()
    assert body["available"] is True and body["n_articles"] == 3
    assert [h["title"] for h in body["up"]] == ["prices soar"]
    assert [h["title"] for h in body["down"]] == ["prices ease"]
    assert body["share_up"] == pytest.approx(1 / 3) and body["share_down"] == pytest.approx(1 / 3)


def test_classify_rejects_empty_headline(client):
    assert client.post("/classify", json={"headline": "  "}).status_code == 422


def test_models_lists_trained_files_and_prefers_news(client):
    body = client.get("/models").json()
    assert [m["name"] for m in body["models"]] == ["linear"] and body["default"] == "linear"
    assert client.get("/nowcast").json()["news"] is None

    add_news_model()
    body = client.get("/models").json()
    assert body["default"] == "linear_news"
    assert {m["name"]: m["uses_news"] for m in body["models"]} == {
        "linear": False, "linear_news": True}


def test_news_model_explains_news_contribution(client):
    add_news_model()
    body = client.get("/nowcast", params={"month": "2021-06"}).json()
    assert body["model"] == "linear_news" and body["uses_news"] is True
    assert "Noticias" in [d["group"] for d in body["drivers"]]
    assert set(body["news"]) == set(params.NEWS_COLS)


def test_comparison_needs_both_models_and_uses_test_months(client):
    assert client.get("/comparison").status_code == 404
    add_news_model()
    body = client.get("/comparison", params={"model": "linear"}).json()
    assert body["n_months"] == params.N_TEST_MONTHS
    assert (body["start"], body["end"]) == ("2020-01", "2023-12")
    assert body["with"]["rmse"] > 0 and body["without"]["rmse"] > 0
    assert 0 <= body["p_value"] <= 1
    # El modelo sin noticias aprende con los mismos meses que el modelo con noticias
    assert (body["train_start"], body["train_end"]) == ("1999-02", "2019-12")


def test_search_headlines_matches_every_word(client):
    assert client.get("/headlines/search", params={"q": "oil"}).json()["available"] is False

    pd.DataFrame({
        "Date": pd.to_datetime(["2021-06-02", "2021-06-10", "2021-07-01", "2021-07-02"], utc=True),
        "Article_title": ["Oil prices soar", "Oil output steady", "Food prices ease", "Oil prices soar"],
        "infla_score": [0.9, 0.1, -0.7, 0.8],
        "infla_role": ["up", "neu", "down", "up"],
    }).to_parquet(params.NEWS_SCORED_PATH)
    fast.load_scored_news.cache_clear()

    body = client.get("/headlines/search", params={"q": "OIL prices"}).json()
    assert body["n_matches"] == 1  # titulares distintos
    assert [r["title"] for r in body["results"]] == ["Oil prices soar"]  # sin repetidos
    assert body["results"][0]["date"] == "2021-07-02"  # el más reciente
    assert client.get("/headlines/search", params={"q": "x"}).status_code == 422


def test_keywords_count_whole_words_and_search_by_keyword(client, tmp_path, monkeypatch):
    config = tmp_path / "model1.yaml"
    config.write_text("keywords_broad: [oil, food prices, wages]\n", encoding="utf-8")
    monkeypatch.setattr(params, "MODEL1_CONFIG_PATH", config)
    assert client.get("/headlines/keywords").json()["available"] is False

    pd.DataFrame({
        "Date": pd.to_datetime(["2021-06-02", "2021-06-10", "2021-07-01"], utc=True),
        "Article_title": ["Oil prices soar", "Toilet paper shortage", "Food prices ease"],
        "infla_score": [0.9, 0.1, -0.7],
        "infla_role": ["up", "neu", "down"],
    }).to_parquet(params.NEWS_SCORED_PATH)
    fast.load_scored_news.cache_clear()

    body = client.get("/headlines/keywords").json()
    assert body["keywords"] == [{"keyword": "oil", "n_headlines": 1},
                                {"keyword": "food prices", "n_headlines": 1}]  # sin "wages" (0)
    found = client.get("/headlines/search", params={"q": "oil", "keyword": True}).json()
    assert [r["title"] for r in found["results"]] == ["Oil prices soar"]  # no "Toilet"
