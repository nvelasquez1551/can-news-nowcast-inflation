"""API del nowcast (FastAPI). Solo lee: el modelo sale del registry y los datos de data/.

    uvicorn nowcast.api.fast:app --reload

Endpoints:
    GET  /                    estado y modelo cargado
    GET  /models              modelos guardados (con y sin noticias) y el modelo por defecto
    GET  /months              meses con nowcast disponible
    GET  /history             dato oficial vs nowcast, todos los meses
    GET  /nowcast?month=      nowcast de un mes y por qué cambió frente al anterior
    GET  /headlines?month=    titulares del mes que más empujan al alza / baja (Modelo 1)
    GET  /headlines/search?q= titulares de la base que contienen esas palabras, con su puntaje
    GET  /comparison          error en 2020-2023 del mismo modelo con y sin noticias
    POST /classify            clasifica un titular con FinBERT e InflaBERT (Modelo 1)

`model` acepta linear, xgb, linear_news y xgb_news. Sin `model`, se usa el modelo
principal con noticias si está entrenado, y si no, el modelo principal macro.
"""

from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from nowcast import params
from nowcast.ml_logic.data import load_news_index, load_raw_data
from nowcast.ml_logic.explain import describe_change, explain_change
from nowcast.ml_logic.features import build_model_table
from nowcast.ml_logic.model import diebold_mariano
from nowcast.ml_logic.registry import load_model

app = FastAPI(title="Nowcast de inflación EE. UU.", version="0.1.0")


def annualize(monthly: float) -> float:
    return 100 * ((1 + monthly / 100) ** 12 - 1)


MODEL_LABELS = {"linear": "Regresión lineal", "xgb": "XGBoost"}


def model_names() -> list[str]:
    return params.MODEL_NAMES + [f"{m}{params.NEWS_MODEL_SUFFIX}" for m in params.MODEL_NAMES]


def available_models() -> list[str]:
    return [name for name in model_names()
            if any(params.MODELS_DIR.glob(f"{name}_[0-9]*.joblib"))]


def resolve_model(model_name: str | None) -> str:
    if model_name is not None:
        return model_name
    with_news = f"{params.PRIMARY_MODEL}{params.NEWS_MODEL_SUFFIX}"
    return with_news if with_news in available_models() else params.PRIMARY_MODEL


@lru_cache
def get_model(model_name: str):
    if model_name not in model_names():
        raise HTTPException(404, f"unknown model '{model_name}', use one of {model_names()}")
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


def model_and_data(model_name: str | None):
    pipe, metadata = get_model(resolve_model(model_name))
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


ModelName = Query(None, description="linear, xgb, linear_news o xgb_news")


@app.get("/")
def root(model: str | None = ModelName):
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


@app.get("/models")
def models():
    names = available_models()
    return {
        "default": resolve_model(None),
        "models": [{"name": name,
                    "label": MODEL_LABELS[name.removesuffix(params.NEWS_MODEL_SUFFIX)],
                    "uses_news": name.endswith(params.NEWS_MODEL_SUFFIX)} for name in names],
    }


@app.get("/months")
def months(model: str | None = ModelName):
    _, _, table, _ = model_and_data(model)
    available = nowcast_months(table)
    return {"months": [f"{m:%Y-%m}" for m in available], "latest": f"{available[-1]:%Y-%m}"}


@app.get("/history")
def history(model: str | None = ModelName):
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
            model: str | None = ModelName):
    pipe, metadata, table, X = model_and_data(model)
    target = parse_month(month, nowcast_months(table))
    explanation = explain_change(pipe, X, target)
    value = explanation["nowcast"]
    previous = explanation["previous_month"]
    news = None
    if all(col in X.columns for col in params.NEWS_COLS):
        news = {col: {"value": float(X.loc[target, col]), "previous": float(X.loc[previous, col])}
                for col in params.NEWS_COLS}
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
        "news": news,
    }


@app.get("/comparison")
def comparison(model: str = Query(params.PRIMARY_MODEL, description="linear o xgb")):
    """Mismo tipo de modelo con y sin noticias, en los meses de test que ambos cubren.

    Son los modelos guardados (entrenados hasta 2019), no el expanding refit.
    """
    base = model.removesuffix(params.NEWS_MODEL_SUFFIX)
    names = {"without": base, "with": f"{base}{params.NEWS_MODEL_SUFFIX}"}
    if not set(names.values()) <= set(available_models()):
        raise HTTPException(404, f"train both {names['without']} and {names['with']} first")

    predictions = {}
    for key, name in names.items():
        pipe, metadata, table, X = model_and_data(name)
        test = table.loc[params.TEST_START:params.TEST_END]
        predictions["y"] = test[params.TARGET]
        predictions[key] = pd.Series(pipe.predict(X.loc[test.index]), index=test.index)
    frame = pd.DataFrame(predictions).dropna()

    def errors(column: str) -> dict:
        error = frame["y"] - frame[column]
        return {"rmse": float((error ** 2).mean() ** 0.5), "mae": float(error.abs().mean())}

    dm = diebold_mariano(frame, [("with", "without")]).iloc[0]
    return {
        "model": base,
        "start": f"{frame.index[0]:%Y-%m}", "end": f"{frame.index[-1]:%Y-%m}",
        "n_months": len(frame),
        "without": errors("without"), "with": errors("with"),
        "dm_stat": float(dm["dm_stat"]), "p_value": float(dm["p_value"]),
    }


@lru_cache
def load_scored_news() -> pd.DataFrame | None:
    if not params.NEWS_SCORED_PATH.exists():
        return None
    news = pd.read_parquet(params.NEWS_SCORED_PATH,
                           columns=["Date", "Article_title", "infla_score", "infla_role"])
    news["month"] = (pd.to_datetime(news["Date"], utc=True)
                     .dt.tz_convert(None).dt.to_period("M").dt.to_timestamp())
    news["title_lower"] = news["Article_title"].astype(str).str.lower()
    return news


def headline_record(row) -> dict:
    return {"date": f"{pd.Timestamp(row.Date):%Y-%m-%d}", "title": row.Article_title,
            "score": float(row.infla_score), "role": str(row.infla_role)}


@app.get("/headlines")
def headlines(month: str = Query(..., description="YYYY-MM"),
              k: int = Query(params.N_TOP_HEADLINES, ge=1, le=20)):
    news = load_scored_news()
    if news is None:
        return {"month": month, "available": False, "up": [], "down": []}
    target = parse_month(month, pd.DatetimeIndex(sorted(news["month"].unique())))
    month_news = news.loc[news["month"] == target]
    rows = month_news.sort_values("infla_score")
    # El mismo titular puede repetirse en días distintos: en las listas va una sola vez
    unique = rows.drop_duplicates("Article_title", keep="last")

    def records(frame: pd.DataFrame) -> list[dict]:
        return [headline_record(row) for row in frame.itertuples()]

    roles = rows["infla_role"].astype(str)
    return {"month": f"{target:%Y-%m}", "available": True, "n_articles": len(rows),
            "share_up": float((roles == "up").mean()), "share_down": float((roles == "down").mean()),
            "up": records(unique.tail(k).iloc[::-1]),
            "down": records(unique.drop_duplicates("Article_title").head(k))}


@app.get("/headlines/search")
def search_headlines(q: str = Query(..., min_length=2, description="una o varias palabras"),
                     k: int = Query(8, ge=1, le=30)):
    """Titulares que contienen todas las palabras de `q`, los más recientes primero."""
    news = load_scored_news()
    if news is None:
        return {"available": False, "n_matches": 0, "results": []}
    mask = pd.Series(True, index=news.index)
    for word in q.lower().split():
        mask &= news["title_lower"].str.contains(word, regex=False)
    hits = (news.loc[mask].sort_values("Date", ascending=False)
            .drop_duplicates("Article_title").head(k))
    return {"available": True, "n_matches": int(mask.sum()),
            "results": [headline_record(row) for row in hits.itertuples()]}


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
