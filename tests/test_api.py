"""Tests de la API con datos y modelo sintéticos (no lee data/ ni models/)."""

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from nowcast import params
from nowcast.api import fast
from nowcast.ml_logic.data import split_by_date
from nowcast.ml_logic.model import build_model


@pytest.fixture
def client(table, monkeypatch, tmp_path):
    train, _ = split_by_date(table)
    pipe = build_model("linear").fit(train[params.FEATURES], train[params.TARGET])
    metadata = {"model_name": "linear", "trained_at": "2026-01-01T00:00:00",
                "train_start": "1999-02", "train_end": "2019-12",
                "features": params.FEATURES}

    def get_model(model_name):
        if model_name not in params.MODEL_NAMES:
            raise fast.HTTPException(404, "unknown model")
        return pipe, metadata

    monkeypatch.setattr(fast, "get_model", get_model)
    monkeypatch.setattr(fast, "get_table", lambda features: table)
    monkeypatch.setattr(params, "NEWS_SCORED_PATH", tmp_path / "news_scored.parquet")
    fast.load_scored_news.cache_clear()
    yield TestClient(fast.app)
    fast.load_scored_news.cache_clear()


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
    }).to_parquet(params.NEWS_SCORED_PATH)
    fast.load_scored_news.cache_clear()

    body = client.get("/headlines", params={"month": "2021-06", "k": 1}).json()
    assert body["available"] is True and body["n_articles"] == 3
    assert [h["title"] for h in body["up"]] == ["prices soar"]
    assert [h["title"] for h in body["down"]] == ["prices ease"]


def test_classify_rejects_empty_headline(client):
    assert client.post("/classify", json={"headline": "  "}).status_code == 422
