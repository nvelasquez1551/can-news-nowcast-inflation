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
    GET  /headlines/keywords  palabras clave del Modelo 1 y cuántos titulares tiene cada una
                              (los tres /headlines* se apagan con HEADLINES_ENABLED=0)
    GET  /comparison          error fuera de muestra (test 2021-01..2026-10) de la lineal con y sin noticias
    POST /classify            clasifica un titular con FinBERT e InflaBERT (Modelo 1);
                              se apaga con CLASSIFY_ENABLED=0

`model` acepta linear y linear_news. Sin `model`, se usa linear_news si está entrenado;
si falta el modelo con noticias o el índice de noticias de un mes, se usa linear (solo
macro) y la respuesta lo dice en `fallback`. Los modelos guardados son los finales
(entrenados con todos los meses): `in_sample` marca los meses que vieron.
"""

import re
from functools import lru_cache

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from nowcast import params
from nowcast.ml_logic.data import load_news_index, load_raw_data, split_by_date
from nowcast.ml_logic.explain import describe_change, explain_change
from nowcast.ml_logic.features import build_model_table
from nowcast.ml_logic.model import build_model, diebold_mariano
from nowcast.ml_logic.registry import load_model

app = FastAPI(title="Nowcast de inflación EE. UU.", version="0.1.0")


def annualize(monthly: float) -> float:
    return 100 * ((1 + monthly / 100) ** 12 - 1)


MODEL_LABELS = {"linear": "Regresión lineal"}
# La API solo sirve la regresión lineal (experimento final). XGBoost sigue en el paquete
# para entrenar y comparar, pero no se expone aquí.
FALLBACK_MODEL = params.PRIMARY_MODEL                                # linear (solo macro)
DEFAULT_MODEL = f"{params.PRIMARY_MODEL}{params.NEWS_MODEL_SUFFIX}"  # linear_news
API_MODELS = [FALLBACK_MODEL, DEFAULT_MODEL]


def model_names() -> list[str]:
    return API_MODELS


def available_models() -> list[str]:
    return [name for name in model_names()
            if any(params.MODELS_DIR.glob(f"{name}_[0-9]*.joblib"))]


def uses_news(features) -> bool:
    return all(col in features for col in params.NEWS_COLS)


def resolve_model(model_name: str | None) -> str:
    """Modelo pedido, o linear_news por defecto si está entrenado (si no, linear)."""
    if model_name is not None:
        return model_name
    return DEFAULT_MODEL if DEFAULT_MODEL in available_models() else FALLBACK_MODEL


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
    news_model = uses_news(features)
    news = load_news_index() if news_model else None
    if news_model and news is None:
        raise HTTPException(503, f"model uses news but {params.NEWS_INDEX_PATH} is missing")
    try:
        raw = load_raw_data()
    except FileNotFoundError as exc:
        raise HTTPException(503, f"{params.RAW_MACRO_PATH} is missing; run make data") from exc
    return build_model_table(raw, news=news, use_news=news_model)


def model_and_data(model_name: str | None):
    """(pipe, metadata, table, X, fallback). Si el modelo con noticias no está entrenado
    o falta el índice de noticias, usa la lineal solo macro y lo dice en `fallback`."""
    name = model_name or DEFAULT_MODEL
    if name not in model_names():
        raise HTTPException(404, f"unknown model '{name}', use one of {model_names()}")
    fallback = None
    if name != FALLBACK_MODEL and name not in available_models():
        fallback = f"{name} is not trained: using macro-only {FALLBACK_MODEL}"
    else:
        pipe, metadata = get_model(name)
        try:
            table = get_table(tuple(metadata["features"]))
        except HTTPException as exc:
            if name == FALLBACK_MODEL or exc.status_code != 503:
                raise
            fallback = f"{exc.detail}: using macro-only {FALLBACK_MODEL}"
    if fallback is not None:
        pipe, metadata = get_model(FALLBACK_MODEL)
        table = get_table(tuple(metadata["features"]))
    return pipe, metadata, table, table[metadata["features"]], fallback


def to_month(value: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(value).to_period("M").to_timestamp()
    except ValueError as exc:
        raise HTTPException(422, f"month must look like YYYY-MM, got '{value}'") from exc


def parse_month(value: str | None, available: pd.DatetimeIndex) -> pd.Timestamp:
    if value is None:
        return available[-1]
    month = to_month(value)
    if month not in available:
        no_cpi = pd.DatetimeIndex(params.TARGET_NAN_MONTHS)
        if month in no_cpi:
            reason = ("no official CPI target for this month (the BLS did not publish the "
                      "October 2025 CPI, and y for 2025-10 and 2025-11 depends on it)")
        elif month - pd.offsets.MonthBegin(1) in no_cpi:
            reason = "the previous month has no CPI target, so the change cannot be explained"
        else:
            reason = f"available {available[0]:%Y-%m} to {available[-1]:%Y-%m}"
        raise HTTPException(404, f"no nowcast for {month:%Y-%m}: {reason}")
    return month


def nowcast_months(table: pd.DataFrame) -> pd.DatetimeIndex:
    """Meses con nowcast y con el mes anterior en la tabla (para explicar el cambio)."""
    previous = table.index - pd.offsets.MonthBegin(1)
    return table.index[previous.isin(table.index)]


def sample_of(month: pd.Timestamp, metadata: dict) -> str:
    """in_sample = el modelo guardado vio ese mes al entrenarse (train_start..train_end)."""
    if month < pd.Timestamp(metadata["train_start"]):
        return "before_training"
    if month > pd.Timestamp(metadata["train_end"]):
        return "out_of_sample"
    return "in_sample"


ModelName = Query(None, description="linear o linear_news (por defecto linear_news)")


@app.get("/")
def root(model: str | None = ModelName):
    _, metadata, table, _, fallback = model_and_data(model)
    return {
        "status": "ok",
        "model": metadata["model_name"],
        "fallback": fallback,
        "trained_at": metadata["trained_at"],
        "train_start": metadata["train_start"],
        "train_end": metadata["train_end"],
        "uses_news": uses_news(metadata["features"]),
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
    _, _, table, _, _ = model_and_data(model)
    available = nowcast_months(table)
    return {"months": [f"{m:%Y-%m}" for m in available], "latest": f"{available[-1]:%Y-%m}"}


@app.get("/history")
def history(model: str | None = ModelName):
    pipe, metadata, table, X, fallback = model_and_data(model)
    rows = pd.DataFrame({
        "month": table.index.strftime("%Y-%m"),
        "actual": table[params.TARGET].to_numpy(),
        "nowcast": pipe.predict(X),
        "in_sample": [sample_of(m, metadata) == "in_sample" for m in table.index],
    })
    return {"model": metadata["model_name"], "fallback": fallback,
            "train_start": metadata["train_start"], "train_end": metadata["train_end"],
            "rows": rows.to_dict(orient="records")}


@app.get("/nowcast")
def nowcast(month: str | None = Query(None, description="YYYY-MM; vacío = último mes"),
            model: str | None = ModelName):
    pipe, metadata, table, X, fallback = model_and_data(model)
    available = nowcast_months(table)
    # Mes sin índice de noticias: se usa la lineal solo macro para ese mes
    if (month is not None and uses_news(metadata["features"])
            and to_month(month) not in available):
        macro = model_and_data(FALLBACK_MODEL)
        if to_month(month) in nowcast_months(macro[2]):
            pipe, metadata, table, X, _ = macro
            available = nowcast_months(table)
            fallback = (f"no news index for {to_month(month):%Y-%m}: "
                        f"using macro-only {FALLBACK_MODEL}")
    target = parse_month(month, available)
    explanation = explain_change(pipe, X, target)
    value = explanation["nowcast"]
    previous = explanation["previous_month"]
    news = None
    if uses_news(X.columns):
        news = {col: {"value": float(X.loc[target, col]), "previous": float(X.loc[previous, col])}
                for col in params.NEWS_COLS}
    sample = sample_of(target, metadata)
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
        "in_sample": sample == "in_sample",
        "sample": sample,
        "train_start": metadata["train_start"],
        "train_end": metadata["train_end"],
        "model": metadata["model_name"],
        "uses_news": uses_news(metadata["features"]),
        "fallback": fallback,
        "news": news,
    }


@lru_cache
def live_comparison() -> dict:
    """linear vs linear_news, re-ajustados AQUÍ con TRAIN_START..TRAIN_END y evaluados en
    los meses de test (fuera de muestra). No usa los modelos finales guardados, que
    aprendieron con todos los meses. Los dos modelos usan exactamente los mismos meses."""
    with_cols = params.FEATURES + params.NEWS_COLS
    table = get_table(tuple(with_cols))
    try:
        train, test = split_by_date(table)
    except AssertionError as exc:
        raise HTTPException(503, f"comparison needs the full test window: {exc}") from exc
    frame = pd.DataFrame({"y": test[params.TARGET]}, index=test.index)
    for key, cols in (("without", params.FEATURES), ("with", with_cols)):
        pipe = build_model(params.PRIMARY_MODEL, cols).fit(train[cols], train[params.TARGET])
        frame[key] = pipe.predict(test[cols])

    def errors(column: str) -> dict:
        error = frame["y"] - frame[column]
        return {"rmse": float((error ** 2).mean() ** 0.5), "mae": float(error.abs().mean())}

    dm = diebold_mariano(frame, [("with", "without")]).iloc[0]
    window = pd.date_range(params.TEST_START, params.TEST_END, freq="MS")
    return {
        "evaluation": "out_of_sample",
        "description": (f"{params.PRIMARY_MODEL} with and without news, both refit on "
                        f"{train.index[0]:%Y-%m}..{train.index[-1]:%Y-%m} and scored on "
                        f"{len(frame)} later months the models did not see. The saved final "
                        "models (trained on all months) are not used here."),
        "model": params.PRIMARY_MODEL,
        "train_start": f"{train.index[0]:%Y-%m}", "train_end": f"{train.index[-1]:%Y-%m}",
        "n_train": len(train),
        "test_start": f"{frame.index[0]:%Y-%m}", "test_end": f"{frame.index[-1]:%Y-%m}",
        "n_months": len(frame),
        "excluded_months": list(window.difference(frame.index).strftime("%Y-%m")),
        "without": errors("without"), "with": errors("with"),
        "loss": "squared error (Diebold-Mariano, h=1, HLN correction)",
        "dm_stat": float(dm["dm_stat"]), "p_value": float(dm["p_value"]),
        "significant_5pct": bool(dm["p_value"] < 0.05),
    }


@app.get("/comparison")
def comparison():
    """Error fuera de muestra de la regresión lineal con y sin noticias (ver live_comparison)."""
    return live_comparison()


def headlines_disabled() -> dict | None:
    """Con HEADLINES_ENABLED=0 los endpoints de titulares responden solo este aviso."""
    if params.HEADLINES_ENABLED:
        return None
    return {"enabled": False, "available": False, "message": params.HEADLINES_DISABLED_MESSAGE}


@lru_cache
def load_scored_news() -> pd.DataFrame | None:
    if not params.NEWS_SCORED_PATH.exists():
        return None
    columns = ["Date", "Article_title", "infla_score", "infla_role"]
    available = pq.read_schema(params.NEWS_SCORED_PATH).names
    news = pd.read_parquet(params.NEWS_SCORED_PATH,
                           columns=columns + (["source"] if "source" in available else []))
    if "source" not in news:
        news["source"] = None
    news["month"] = (pd.to_datetime(news["Date"], utc=True)
                     .dt.tz_convert(None).dt.to_period("M").dt.to_timestamp())
    news["title_lower"] = news["Article_title"].astype(str).str.lower()
    return news


def headline_record(row) -> dict:
    return {"date": f"{pd.Timestamp(row.Date):%Y-%m-%d}", "title": row.Article_title,
            "score": float(row.infla_score), "role": str(row.infla_role),
            "source": row.source}


@app.get("/headlines")
def headlines(month: str = Query(..., description="YYYY-MM"),
              k: int = Query(params.N_TOP_HEADLINES, ge=1, le=20)):
    if disabled := headlines_disabled():
        return {"month": month, **disabled, "up": [], "down": []}
    news = load_scored_news()
    if news is None:
        return {"month": month, "enabled": True, "available": False, "up": [], "down": []}
    target = parse_month(month, pd.DatetimeIndex(sorted(news["month"].unique())))
    month_news = news.loc[news["month"] == target]
    rows = month_news.sort_values("infla_score")
    # El mismo titular puede repetirse en días distintos: en las listas va una sola vez
    unique = rows.drop_duplicates("Article_title", keep="last")

    def records(frame: pd.DataFrame) -> list[dict]:
        return [headline_record(row) for row in frame.itertuples()]

    roles = rows["infla_role"].astype(str)
    return {"month": f"{target:%Y-%m}", "enabled": True, "available": True,
            "n_articles": len(rows),
            "sources": sorted(rows["source"].dropna().unique().tolist()),
            "share_up": float((roles == "up").mean()), "share_down": float((roles == "down").mean()),
            "up": records(unique.tail(k).iloc[::-1]),
            "down": records(unique.drop_duplicates("Article_title").head(k))}


@app.get("/headlines/search")
def search_headlines(q: str = Query(..., min_length=2, description="una o varias palabras"),
                     k: int = Query(8, ge=1, le=30),
                     keyword: bool = Query(False, description="q es una palabra clave exacta"),
                     month: str | None = Query(None, description="YYYY-MM: solo ese mes")):
    """Titulares que contienen todas las palabras de `q` (completas), los más recientes primero.

    Con keyword=true, `q` se busca como frase completa ("oil" no encuentra "toilet").
    Con `month`, solo busca en los titulares de ese mes.
    """
    if disabled := headlines_disabled():
        return {**disabled, "n_matches": 0, "results": []}
    news = load_scored_news()
    if news is None:
        return {"enabled": True, "available": False, "n_matches": 0, "results": []}
    if month is not None:
        target = parse_month(month, pd.DatetimeIndex(sorted(news["month"].unique())))
        news = news.loc[news["month"] == target]
    mask = pd.Series(True, index=news.index)
    if keyword:
        mask &= contains_keyword(news["title_lower"], q)
    else:
        # Palabras completas: "oil" no encuentra "Spoil"
        for word in q.lower().split():
            mask &= contains_keyword(news["title_lower"], word)
    hits = (news.loc[mask].sort_values("Date", ascending=False)
            .drop_duplicates("Article_title").head(k))
    return {"enabled": True, "available": True,
            "n_matches": int(news.loc[mask, "title_lower"].nunique()),
            "results": [headline_record(row) for row in hits.itertuples()]}


def contains_keyword(titles: pd.Series, keyword: str) -> pd.Series:
    """Palabra o frase completa ("oil" no encuentra "toilet"). pyarrow: ~10x más rápido que pandas."""
    pattern = rf"\b{re.escape(keyword.strip().lower())}\b"
    hits = pc.match_substring_regex(pa.array(titles.to_numpy(dtype=object)), pattern)
    return pd.Series(hits.to_numpy(zero_copy_only=False), index=titles.index)


@lru_cache
def keyword_counts() -> list[dict]:
    news = load_scored_news()
    config = yaml.safe_load(params.MODEL1_CONFIG_PATH.read_text(encoding="utf-8"))
    titles = news["title_lower"].drop_duplicates()
    counts = [{"keyword": kw, "n_headlines": int(contains_keyword(titles, kw).sum())}
              for kw in config["keywords_broad"]]
    return sorted([c for c in counts if c["n_headlines"]], key=lambda c: -c["n_headlines"])


@app.get("/headlines/keywords")
def headline_keywords():
    """Palabras clave del Modelo 1 con el número de titulares distintos que las contienen."""
    if disabled := headlines_disabled():
        return {**disabled, "keywords": []}
    if load_scored_news() is None:
        return {"enabled": True, "available": False, "keywords": []}
    return {"enabled": True, "available": True, "keywords": keyword_counts()}


class Headline(BaseModel):
    headline: str


@app.post("/classify")
def classify(body: Headline):
    """Puntaje raw de FinBERT e InflaBERT para un titular. Apagado con CLASSIFY_ENABLED=0."""
    if not params.CLASSIFY_ENABLED:
        return {"enabled": False, "message": params.CLASSIFY_DISABLED_MESSAGE}
    if not body.headline.strip():
        raise HTTPException(422, "headline must not be empty")
    try:
        from src.model1.predict import classify_headline
    except ImportError as exc:
        raise HTTPException(503, f"Model 1 needs torch and transformers: {exc}") from exc
    try:
        result = classify_headline(body.headline)
    except Exception as exc:  # pesos que no se descargan, memoria, etc.
        raise HTTPException(503, f"Model 1 could not be loaded: {exc}") from exc
    return {"enabled": True, **result}
