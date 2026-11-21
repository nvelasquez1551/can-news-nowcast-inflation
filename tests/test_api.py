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
    # Como el índice real, las noticias empiezan en TRAIN_START (antes: solo macro)
    tables = {"macro": table, "news": news_table(table).loc[params.TRAIN_START:]}
    fitted = {}
    for name, cols, key in (("linear", params.FEATURES, "macro"),
                            ("linear_news", params.FEATURES + params.NEWS_COLS, "news")):
        train, _ = split_by_date(tables[key])
        pipe = build_model("linear", cols).fit(train[cols], train[params.TARGET])
        fitted[name] = (pipe, {"model_name": name, "trained_at": "2026-01-01T00:00:00",
                               "train_start": f"{pd.Timestamp(params.TRAIN_START):%Y-%m}",
                               "train_end": f"{pd.Timestamp(params.TRAIN_END):%Y-%m}",
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
    monkeypatch.setattr(params, "HEADLINES_ENABLED", True)  # no depende del entorno
    monkeypatch.setattr(params, "CLASSIFY_ENABLED", True)
    caches = (fast.load_scored_news, fast.keyword_counts, fast.live_comparison,
              fast.load_news_stats)
    for cached in caches:
        cached.cache_clear()
    yield TestClient(fast.app)
    for cached in caches:
        cached.cache_clear()


def add_news_model():
    (params.MODELS_DIR / "linear_news_20260101-000000-000000.joblib").touch()


def test_root_and_months(client, table):
    root = client.get("/").json()
    assert root["status"] == "ok" and root["uses_news"] is False
    months = client.get("/months").json()
    assert months["latest"] == f"{table.index[-1]:%Y-%m}"
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


def test_nowcast_defaults_to_latest_and_validates_month(client, table):
    assert client.get("/nowcast").json()["month"] == f"{table.index[-1]:%Y-%m}"
    assert client.get("/nowcast", params={"month": "abc"}).status_code == 422
    assert client.get("/nowcast", params={"month": "1990-01"}).status_code == 404
    assert client.get("/nowcast", params={"model": "rf"}).status_code == 404


def test_history_matches_table(client, table):
    rows = client.get("/history").json()["rows"]
    assert len(rows) == len(table)
    assert rows[-1]["month"] == f"{table.index[-1]:%Y-%m}" and rows[-1]["in_sample"] is False
    by_month = {r["month"]: r["in_sample"] for r in rows}
    assert by_month[f"{pd.Timestamp(params.TRAIN_START):%Y-%m}"] is True
    assert rows[0]["in_sample"] is False  # antes de TRAIN_START: el modelo no lo vio


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


def test_comparison_refits_live_out_of_sample(client, table, monkeypatch):
    # No usa los modelos guardados: funciona aunque no haya ninguno
    def no_saved_models(name):
        raise AssertionError("comparison must not load saved models")
    monkeypatch.setattr(fast, "get_model", no_saved_models)
    body = client.get("/comparison").json()

    month = lambda value: f"{pd.Timestamp(value):%Y-%m}"
    assert body["evaluation"] == "out_of_sample" and body["model"] == "linear"
    assert (body["train_start"], body["train_end"]) == (month(params.TRAIN_START),
                                                        month(params.TRAIN_END))
    assert (body["test_start"], body["test_end"]) == (month(params.TEST_START),
                                                      month(params.TEST_END))
    assert body["n_months"] == params.N_TEST_MONTHS
    assert body["excluded_months"] == [month(m) for m in params.TARGET_NAN_MONTHS]

    # Mismo resultado que ajustar a mano las dos lineales con los mismos meses de train
    train, test = split_by_date(news_table(table).loc[params.TRAIN_START:])
    for key, cols in (("without", params.FEATURES),
                      ("with", params.FEATURES + params.NEWS_COLS)):
        pipe = build_model("linear", cols).fit(train[cols], train[params.TARGET])
        error = test[params.TARGET] - pipe.predict(test[cols])
        assert body[key]["rmse"] == pytest.approx(np.sqrt((error ** 2).mean()))
        assert body[key]["mae"] == pytest.approx(error.abs().mean())
    assert 0 <= body["p_value"] <= 1
    assert body["significant_5pct"] == (body["p_value"] < 0.05)


def test_default_model_is_linear_news_with_fallbacks(client, monkeypatch):
    # Sin linear_news entrenado: lineal macro y se avisa
    body = client.get("/nowcast").json()
    assert body["model"] == "linear" and "linear_news is not trained" in body["fallback"]

    add_news_model()
    body = client.get("/nowcast").json()
    assert body["model"] == "linear_news" and body["fallback"] is None

    # Mes sin índice de noticias (antes de TRAIN_START): ese mes usa la lineal macro
    early = f"{pd.Timestamp(params.TRAIN_START) - pd.DateOffset(years=2):%Y-%m}"
    body = client.get("/nowcast", params={"month": early}).json()
    assert body["model"] == "linear" and body["news"] is None
    assert f"no news index for {early}" in body["fallback"]

    # Falta el archivo del índice de noticias: todo cae a la lineal macro
    tables_get = fast.get_table
    def missing_news(features):
        if params.NEWS_COLS[0] in features:
            raise fast.HTTPException(503, "model uses news but the news index is missing")
        return tables_get(features)
    monkeypatch.setattr(fast, "get_table", missing_news)
    body = client.get("/nowcast").json()
    assert body["model"] == "linear" and "news index is missing" in body["fallback"]


def test_xgb_is_not_served(client):
    (params.MODELS_DIR / "xgb_20260101-000000-000000.joblib").touch()
    assert "xgb" not in [m["name"] for m in client.get("/models").json()["models"]]
    assert client.get("/nowcast", params={"model": "xgb"}).status_code == 404


def test_in_sample_follows_saved_model_training_window(client):
    month = lambda value: f"{pd.Timestamp(value):%Y-%m}"
    seen = client.get("/nowcast", params={"month": month(params.TRAIN_END)}).json()
    assert seen["in_sample"] is True and seen["sample"] == "in_sample"
    after = month(pd.Timestamp(params.TRAIN_END) + pd.DateOffset(months=6))
    assert client.get("/nowcast", params={"month": after}).json()["sample"] == "out_of_sample"
    before = month(pd.Timestamp(params.TRAIN_START) - pd.DateOffset(years=2))
    assert client.get("/nowcast", params={"month": before}).json()["sample"] == "before_training"
    rows = pd.DataFrame(client.get("/history").json()["rows"])
    inside = rows["month"].between(month(params.TRAIN_START), month(params.TRAIN_END))
    assert (rows["in_sample"] == inside).all()


def test_months_without_cpi_do_not_crash(client):
    for month in ("2025-10", "2025-11"):
        response = client.get("/nowcast", params={"month": month})
        assert response.status_code == 404 and "no official CPI" in response.json()["detail"]
    response = client.get("/nowcast", params={"month": "2025-12"})
    assert response.status_code == 404 and "previous month" in response.json()["detail"]
    months = client.get("/months").json()["months"]
    assert not {"2025-10", "2025-11", "2025-12"} & set(months)
    history = [r["month"] for r in client.get("/history").json()["rows"]]
    assert not {"2025-10", "2025-11"} & set(history)


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


def write_scored_news_with_sources():
    pd.DataFrame({
        "Date": pd.to_datetime(["2014-12-05", "2021-06-02", "2021-06-10", "2021-07-01"], utc=True),
        "Article_title": ["Oil prices fall", "Oil prices soar", "Food prices ease", "Oil prices up"],
        "infla_score": [-0.6, 0.9, -0.7, 0.5],
        "infla_role": ["down", "up", "down", "up"],
        "source": ["fnspid", "alpaca", "alpaca", "alpaca"],
    }).to_parquet(params.NEWS_SCORED_PATH)
    fast.load_scored_news.cache_clear()


def test_headlines_carry_source_and_search_by_month(client):
    write_scored_news_with_sources()
    body = client.get("/headlines", params={"month": "2014-12"}).json()
    assert body["sources"] == ["fnspid"] and body["down"][0]["source"] == "fnspid"
    found = client.get("/headlines/search", params={"q": "oil", "month": "2021-06"}).json()
    assert [r["title"] for r in found["results"]] == ["Oil prices soar"]
    assert found["n_matches"] == 1 and found["results"][0]["source"] == "alpaca"
    assert len(client.get("/headlines/search", params={"q": "oil"}).json()["results"]) == 3
    assert client.get("/headlines/search", params={"q": "oil", "month": "1990-01"}).status_code == 404


def test_headlines_without_source_column_still_work(client):
    pd.DataFrame({"Date": pd.to_datetime(["2021-06-02"], utc=True), "Article_title": ["x y"],
                  "infla_score": [0.1], "infla_role": ["neu"]}).to_parquet(params.NEWS_SCORED_PATH)
    fast.load_scored_news.cache_clear()
    assert client.get("/headlines", params={"month": "2021-06"}).json()["up"][0]["source"] is None


def test_headlines_flag_turns_off_the_three_endpoints(client, monkeypatch):
    write_scored_news_with_sources()
    monkeypatch.setattr(params, "HEADLINES_ENABLED", False)
    responses = [client.get("/headlines", params={"month": "2021-06"}),
                 client.get("/headlines/search", params={"q": "oil"}),
                 client.get("/headlines/keywords")]
    for response in responses:
        body = response.json()
        assert response.status_code == 200
        assert body["enabled"] is False and body["available"] is False
        assert body["message"] == "headlines are not available in this deployment"
    assert responses[0].json()["up"] == [] and responses[1].json()["results"] == []
    assert responses[2].json()["keywords"] == []

    monkeypatch.setattr(params, "HEADLINES_ENABLED", True)
    assert client.get("/headlines", params={"month": "2021-06"}).json()["enabled"] is True


@pytest.mark.parametrize("value, enabled", [(None, True), ("1", True), ("true", True),
                                            ("0", False), ("false", False), ("OFF", False)])
def test_headlines_enabled_env_var(monkeypatch, value, enabled):
    import importlib
    if value is None:
        monkeypatch.delenv("HEADLINES_ENABLED", raising=False)
    else:
        monkeypatch.setenv("HEADLINES_ENABLED", value)
    try:
        assert importlib.reload(params).HEADLINES_ENABLED is enabled
    finally:
        monkeypatch.delenv("HEADLINES_ENABLED", raising=False)
        importlib.reload(params)


def test_search_free_text_matches_whole_words(client):
    pd.DataFrame({
        "Date": pd.to_datetime(["2021-06-02", "2021-06-10", "2021-06-20"], utc=True),
        "Article_title": ["Tariffs Spoil the Party", "Oil prices soar", "Boiling oil-market fears"],
        "infla_score": [0.1, 0.9, 0.4],
        "infla_role": ["neu", "up", "up"],
    }).to_parquet(params.NEWS_SCORED_PATH)
    fast.load_scored_news.cache_clear()
    titles = [r["title"] for r in client.get("/headlines/search", params={"q": "oil"}).json()["results"]]
    assert titles == ["Boiling oil-market fears", "Oil prices soar"]  # no "Spoil"
    assert client.get("/headlines/search", params={"q": "oi"}).json()["n_matches"] == 0


def fake_model1(monkeypatch, classify_headline):
    import sys
    import types
    module = types.ModuleType("src.model1.predict")
    module.classify_headline = classify_headline
    monkeypatch.setitem(sys.modules, "src.model1.predict", module)


def test_classify_flag_and_errors(client, monkeypatch):
    infla = {"label": "positive", "role": "up", "p_up": 0.9, "p_down": 0.05, "p_neu": 0.05,
             "score": 0.85}
    fake_model1(monkeypatch, lambda text: {"text": text, "inflabert": infla})
    monkeypatch.setattr(params, "CLASSIFY_ENABLED", True)
    body = client.post("/classify", json={"headline": "Gas prices jump"}).json()
    assert body["enabled"] is True and body["inflabert"]["role"] == "up"

    def broken(text):
        raise OSError("cannot download MAPAi/InflaBERT")
    fake_model1(monkeypatch, broken)
    response = client.post("/classify", json={"headline": "Gas prices jump"})
    assert response.status_code == 503 and "could not be loaded" in response.json()["detail"]

    monkeypatch.setattr(params, "CLASSIFY_ENABLED", False)
    response = client.post("/classify", json={"headline": "Gas prices jump"})
    assert response.status_code == 200
    assert response.json() == {"enabled": False,
                               "message": "headline classification is not available in this deployment"}


@pytest.mark.parametrize("value, enabled", [(None, True), ("1", True), ("yes", True),
                                            ("0", False), ("no", False), ("False", False)])
def test_classify_enabled_env_var(monkeypatch, value, enabled):
    import importlib
    if value is None:
        monkeypatch.delenv("CLASSIFY_ENABLED", raising=False)
    else:
        monkeypatch.setenv("CLASSIFY_ENABLED", value)
    try:
        assert importlib.reload(params).CLASSIFY_ENABLED is enabled
    finally:
        monkeypatch.delenv("CLASSIFY_ENABLED", raising=False)
        importlib.reload(params)


@pytest.fixture
def news_index_csv(tmp_path, monkeypatch):
    """Índice de noticias sintético: 2014-11..2015-02, con cambio de fuente en 2015-01."""
    path = tmp_path / "news_index.csv"
    pd.DataFrame({
        "month": ["2014-11-01", "2014-12-01", "2015-01-01", "2015-02-01"],
        "infla_score_mean": [0.10, 0.25, 0.05, 0.20],
        "news_n_articles": [300, 320, 25, 140],
        "source": ["fnspid", "fnspid", "alpaca", "alpaca"],
        "below_min_flag": [False, False, True, False],
    }).to_csv(path, index=False)
    monkeypatch.setattr(params, "NEWS_INDEX_PATH", path)
    fast.load_news_stats.cache_clear()
    return path


def test_news_stats_month_and_previous(client, news_index_csv):
    body = client.get("/news-stats", params={"month": "2015-01"}).json()
    assert body["available"] is True and body["month"] == "2015-01"
    assert body["n_articles"] == 25 and body["source"] == "alpaca" and body["below_min"] is True
    assert body["infla_score_mean"] == pytest.approx(0.05)
    assert body["previous"] == {"month": "2014-12", "n_articles": 320, "source": "fnspid",
                                "below_min": False, "infla_score_mean": pytest.approx(0.25)}
    assert body["score_change"] == pytest.approx(-0.20)
    assert body["source_changed"] is True
    assert [row["month"] for row in body["series"]] == ["2014-11", "2014-12", "2015-01", "2015-02"]
    assert body["first_month"] == "2014-11" and body["last_month"] == "2015-02"


def test_news_stats_first_month_has_no_previous(client, news_index_csv):
    body = client.get("/news-stats", params={"month": "2014-11"}).json()
    assert body["available"] is True and body["previous"] is None
    assert body["score_change"] is None and body["source_changed"] is False


@pytest.mark.parametrize("month, text", [("2006-12", "starts in 2014-11"),
                                         ("2015-03", "ends in 2015-02")])
def test_news_stats_outside_the_index(client, news_index_csv, month, text):
    body = client.get("/news-stats", params={"month": month}).json()
    assert body["available"] is False and text in body["message"]
    assert len(body["series"]) == 4


def test_news_stats_without_index_file(client, tmp_path, monkeypatch):
    monkeypatch.setattr(params, "NEWS_INDEX_PATH", tmp_path / "missing.csv")
    fast.load_news_stats.cache_clear()
    body = client.get("/news-stats", params={"month": "2015-01"}).json()
    assert body["available"] is False and "not available" in body["message"]


def test_news_stats_works_with_headlines_off_and_reads_no_parquet(client, news_index_csv,
                                                                   monkeypatch):
    monkeypatch.setattr(params, "HEADLINES_ENABLED", False)

    def no_parquet(*args, **kwargs):
        raise AssertionError("/news-stats must not read parquet")

    monkeypatch.setattr(fast.pd, "read_parquet", no_parquet)
    monkeypatch.setattr(fast.pq, "read_schema", no_parquet)
    body = client.get("/news-stats", params={"month": "2015-02"}).json()
    assert body["available"] is True and body["n_articles"] == 140
    assert "title" not in str(body).lower()


def test_news_stats_bad_month(client, news_index_csv):
    assert client.get("/news-stats", params={"month": "junio"}).status_code == 422
