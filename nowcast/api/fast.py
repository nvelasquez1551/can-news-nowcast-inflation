"""API del nowcast (FastAPI). Solo lee: el modelo sale del registry y los datos de data/.

    uvicorn nowcast.api.fast:app --reload

Endpoints:
    GET  /                    estado y modelo cargado
    GET  /months              meses con nowcast disponible
    GET  /history             dato oficial vs nowcast, todos los meses
    GET  /nowcast?month=      nowcast de un mes y por qué cambió frente al anterior
    GET  /headlines?month=    titulares del mes que más empujan al alza / baja (Modelo 1)
    POST /classify            clasifica un titular con FinBERT e InflaBERT (Modelo 1)
"""

from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from nowcast import params
from nowcast.ml_logic.data import load_news_index, load_raw_data
from nowcast.ml_logic.explain import describe_change, explain_change
from nowcast.ml_logic.features import build_model_table
from nowcast.ml_logic.registry import load_model

app = FastAPI(title="Nowcast de inflación EE. UU.", version="0.1.0")


def annualize(monthly: float) -> float:
    return 100 * ((1 + monthly / 100) ** 12 - 1)


@lru_cache
def get_model(model_name: str):
    if model_name not in params.MODEL_NAMES:
        raise HTTPException(404, f"unknown model '{model_name}', use one of {params.MODEL_NAMES}")
    try:
        return load_model(model_name)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc


@lru_cache
def get_table(features: tuple[str, ...]) -> pd.DataFrame:
    """Tabla del modelo con las features con las que se entrenó (con noticias si las usa)."""
    uses_news = all(col in features for col in params.NEWS_COLS)
    news = load_news_index() if uses_news else None
    if uses_news and news is None:
        raise HTTPException(503, f"model uses news but {params.NEWS_INDEX_PATH} is missing")
    try:
        raw = load_raw_data()
    except FileNotFoundError as exc:
        raise HTTPException(503, f"{params.RAW_MACRO_PATH} is missing; run make data") from exc
    return build_model_table(raw, news=news, use_news=uses_news)


def model_and_data(model_name: str):
    pipe, metadata = get_model(model_name)
    table = get_table(tuple(metadata["features"]))
    return pipe, metadata, table, table[metadata["features"]]


def parse_month(value: str | None, available: pd.DatetimeIndex) -> pd.Timestamp:
    if value is None:
        return available[-1]
    try:
        month = pd.Timestamp(value).to_period("M").to_timestamp()
    except ValueError as exc:
        raise HTTPException(422, f"month must look like YYYY-MM, got '{value}'") from exc
    if month not in available:
        raise HTTPException(404, f"no nowcast for {month:%Y-%m}; "
                                 f"available {available[0]:%Y-%m} to {available[-1]:%Y-%m}")
    return month


def nowcast_months(table: pd.DataFrame) -> pd.DatetimeIndex:
    """Meses con nowcast y con el mes anterior en la tabla (para explicar el cambio)."""
    previous = table.index - pd.offsets.MonthBegin(1)
    return table.index[previous.isin(table.index)]


ModelName = Query(params.PRIMARY_MODEL, description="linear o xgb")


@app.get("/")
def root(model: str = ModelName):
    _, metadata, table, _ = model_and_data(model)
    return {
        "status": "ok",
        "model": metadata["model_name"],
        "trained_at": metadata["trained_at"],
        "train_start": metadata["train_start"],
        "train_end": metadata["train_end"],
        "uses_news": all(col in metadata["features"] for col in params.NEWS_COLS),
        "first_month": f"{table.index[0]:%Y-%m}",
        "last_month": f"{table.index[-1]:%Y-%m}",
    }


@app.get("/months")
def months(model: str = ModelName):
    _, _, table, _ = model_and_data(model)
    available = nowcast_months(table)
    return {"months": [f"{m:%Y-%m}" for m in available], "latest": f"{available[-1]:%Y-%m}"}


@app.get("/history")
def history(model: str = ModelName):
    pipe, metadata, table, X = model_and_data(model)
    train_end = pd.Timestamp(metadata["train_end"])
    rows = pd.DataFrame({
        "month": table.index.strftime("%Y-%m"),
        "actual": table[params.TARGET].to_numpy(),
        "nowcast": pipe.predict(X),
        "in_sample": (table.index <= train_end),
    })
    return {"model": metadata["model_name"], "train_end": metadata["train_end"],
            "rows": rows.to_dict(orient="records")}


@app.get("/nowcast")
def nowcast(month: str | None = Query(None, description="YYYY-MM; vacío = último mes"),
            model: str = ModelName):
    pipe, metadata, table, X = model_and_data(model)
    target = parse_month(month, nowcast_months(table))
    explanation = explain_change(pipe, X, target)
    value = explanation["nowcast"]
    return {
        "month": f"{target:%Y-%m}",
        "nowcast": value,
        "nowcast_annualized": annualize(value),
        "actual": float(table.loc[target, params.TARGET]),
        "previous_month": f"{explanation['previous_month']:%Y-%m}",
        "previous_nowcast": explanation["previous_nowcast"],
        "change": explanation["change"],
        "base": explanation["base"],
        "summary": describe_change(explanation),
        "drivers": explanation["drivers"].to_dict(orient="records"),
        "in_sample": bool(target <= pd.Timestamp(metadata["train_end"])),
        "model": metadata["model_name"],
        "uses_news": all(col in metadata["features"] for col in params.NEWS_COLS),
    }


@lru_cache
def load_scored_news() -> pd.DataFrame | None:
    if not params.NEWS_SCORED_PATH.exists():
        return None
    news = pd.read_parquet(params.NEWS_SCORED_PATH,
                           columns=["Date", "Article_title", "infla_score"])
    news["month"] = (pd.to_datetime(news["Date"], utc=True)
                     .dt.tz_convert(None).dt.to_period("M").dt.to_timestamp())
    return news


@app.get("/headlines")
def headlines(month: str = Query(..., description="YYYY-MM"),
              k: int = Query(params.N_TOP_HEADLINES, ge=1, le=20)):
    news = load_scored_news()
    if news is None:
        return {"month": month, "available": False, "up": [], "down": []}
    target = parse_month(month, pd.DatetimeIndex(sorted(news["month"].unique())))
    rows = news.loc[news["month"] == target].sort_values("infla_score")

    def records(frame: pd.DataFrame) -> list[dict]:
        return [{"date": f"{pd.Timestamp(r.Date):%Y-%m-%d}", "title": r.Article_title,
                 "score": float(r.infla_score)} for r in frame.itertuples()]

    return {"month": f"{target:%Y-%m}", "available": True, "n_articles": len(rows),
            "up": records(rows.tail(k).iloc[::-1]), "down": records(rows.head(k))}


class Headline(BaseModel):
    headline: str


@app.post("/classify")
def classify(body: Headline):
    if not body.headline.strip():
        raise HTTPException(422, "headline must not be empty")
    try:
        from src.model1.predict import classify_headline
    except ImportError as exc:
        raise HTTPException(503, f"Model 1 needs torch and transformers: {exc}") from exc
    return classify_headline(body.headline)
