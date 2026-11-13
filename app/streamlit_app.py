"""Frontend: nowcast del IPC y por qué cambia, y el clasificador de titulares.

Toda la lógica vive en la API (nowcast/api/fast.py); aquí solo se llama y se grafica.
Desde la raíz del repo, con la API corriendo:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
MODELS = {"linear": "Regresión lineal (principal)", "xgb": "XGBoost"}
EXAMPLES = [
    "Gasoline prices soar as crude rallies",
    "Consumer prices fall for a second month",
    "Fed holds rates steady",
]
ROLE_TEXT = {"up": "Sube", "down": "Baja", "neu": "Neutral", "pos": "Positivo", "neg": "Negativo"}
UP_COLOR, DOWN_COLOR = "#c0392b", "#2e86c1"

st.set_page_config(page_title="Nowcast de inflación", layout="wide")


@st.cache_data(ttl=60, show_spinner=False)
def api_get(path: str, **query) -> dict:
    response = requests.get(f"{API_URL}{path}", params=query, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(response.json().get("detail", response.text))
    return response.json()


def api_post(path: str, payload: dict) -> dict:
    response = requests.post(f"{API_URL}{path}", json=payload, timeout=300)
    if response.status_code != 200:
        raise RuntimeError(response.json().get("detail", response.text))
    return response.json()


def history_chart(history: pd.DataFrame, month: str, train_end: str) -> alt.Chart:
    long = history.melt(id_vars=["month"], value_vars=["actual", "nowcast"],
                        var_name="serie", value_name="valor")
    long["serie"] = long["serie"].map({"actual": "Dato oficial", "nowcast": "Nowcast"})
    base = alt.Chart(long).encode(
        x=alt.X("month:T", title=None),
        y=alt.Y("valor:Q", title="IPC, variación mensual (%)"),
        color=alt.Color("serie:N", title=None,
                        scale=alt.Scale(range=["#7f8c8d", UP_COLOR])),
        tooltip=[alt.Tooltip("month:T", format="%Y-%m", title="Mes"),
                 alt.Tooltip("serie:N", title="Serie"),
                 alt.Tooltip("valor:Q", format=".2f", title="%")],
    )
    lines = base.mark_line(strokeWidth=1.5)
    selected = alt.Chart(pd.DataFrame({"month": [pd.Timestamp(month)]})).mark_rule(
        color="black", strokeDash=[4, 3]).encode(x="month:T")
    test_zone = alt.Chart(pd.DataFrame({
        "start": [pd.Timestamp(train_end) + pd.offsets.MonthBegin(1)],
        "end": [history["month"].max()],
    })).mark_rect(opacity=0.08, color="black").encode(x="start:T", x2="end:T")
    return (test_zone + lines + selected).properties(height=320)


def drivers_chart(drivers: pd.DataFrame, value: str, title: str) -> alt.Chart:
    data = drivers.assign(direction=drivers[value].ge(0).map({True: "Empuja al alza",
                                                              False: "Empuja a la baja"}))
    return alt.Chart(data).mark_bar().encode(
        x=alt.X(f"{value}:Q", title="puntos porcentuales"),
        y=alt.Y("group:N", sort=alt.EncodingSortField(field=value, op="sum", order="descending"),
                title=None, axis=alt.Axis(labelLimit=220)),
        color=alt.Color("direction:N", title=None,
                        scale=alt.Scale(domain=["Empuja al alza", "Empuja a la baja"],
                                        range=[UP_COLOR, DOWN_COLOR])),
        tooltip=[alt.Tooltip("group:N", title="Factor"),
                 alt.Tooltip(f"{value}:Q", format="+.3f", title="pp")],
    ).properties(title=title, height=alt.Step(28))


def show_headlines(month: str) -> None:
    news = api_get("/headlines", month=month)
    if not news["available"]:
        st.caption("Titulares del mes: disponibles cuando exista `data/processed/news_scored.parquet` "
                   "(salida del Modelo 1).")
        return
    st.subheader(f"Titulares de {month} ({news['n_articles']} noticias de precios)")
    up_col, down_col = st.columns(2)
    for column, key, label in ((up_col, "up", "Más inflacionarios"),
                               (down_col, "down", "Más desinflacionarios")):
        with column:
            st.markdown(f"**{label}**")
            for item in news[key]:
                st.markdown(f"- {item['title']} · `{item['score']:+.2f}` · {item['date']}")


def nowcast_tab(model: str) -> None:
    months = api_get("/months", model=model)
    month = st.select_slider("Mes", options=months["months"], value=months["latest"])
    result = api_get("/nowcast", month=month, model=model)
    history = pd.DataFrame(api_get("/history", model=model)["rows"])
    history["month"] = pd.to_datetime(history["month"])
    if not st.toggle("Ver toda la historia (desde 1999)"):
        start = min(pd.Timestamp(month), pd.Timestamp("2016-01-01"))
        history = history[history["month"] >= start]
    train_end = api_get("/", model=model)["train_end"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nowcast (mensual)", f"{result['nowcast']:.2f} %",
                f"{result['change']:+.2f} pp vs {result['previous_month']}", delta_color="inverse")
    col2.metric("Nowcast anualizado", f"{result['nowcast_annualized']:.1f} %")
    col3.metric("Dato oficial", f"{result['actual']:.2f} %")
    col4.metric("Error", f"{result['nowcast'] - result['actual']:+.2f} pp")
    st.info(result["summary"])
    if result["in_sample"]:
        st.caption("Mes dentro del periodo de entrenamiento: el modelo ya vio este dato.")
    if not result["uses_news"]:
        st.caption("Este modelo aún no usa el índice de noticias del Modelo 1.")

    drivers = pd.DataFrame(result["drivers"])
    left, right = st.columns([3, 2])
    with left:
        st.altair_chart(history_chart(history, month, train_end), width="stretch")
        st.caption("Zona sombreada: meses fuera del entrenamiento (desde "
                   f"{pd.Timestamp(train_end) + pd.offsets.MonthBegin(1):%Y-%m}).")
    with right:
        st.altair_chart(drivers_chart(drivers, "change",
                                      f"¿Por qué cambió frente a {result['previous_month']}?"),
                        width="stretch")
    with st.expander("Aporte de cada factor al nivel del nowcast"):
        st.altair_chart(drivers_chart(drivers, "contribution",
                                      f"Nowcast = {result['base']:.2f} (base) + aportes"),
                        width="stretch")
    show_headlines(month)


def show_classification(column, title: str, result: dict, prob_keys: dict[str, str]) -> None:
    with column:
        st.subheader(title)
        st.metric("Etiqueta", ROLE_TEXT.get(result["role"], result["label"]))
        st.metric("Puntaje (-1 a 1)", f"{result['score']:+.2f}")
        probs = pd.Series({ROLE_TEXT[role]: result[key] for role, key in prob_keys.items()})
        st.bar_chart(probs, y_label="Probabilidad")


def classifier_tab() -> None:
    st.caption("InflaBERT mide si el titular implica que los precios suben o bajan. "
               "FinBERT mide si es una noticia buena o mala para los mercados.")
    example = st.selectbox("Ejemplo", ["", *EXAMPLES])
    headline = st.text_input("Titular (en inglés)", value=example)
    if not headline.strip():
        return
    with st.spinner("Clasificando (la primera vez carga los modelos)..."):
        result = api_post("/classify", {"headline": headline})
    infla_col, finbert_col = st.columns(2)
    show_classification(infla_col, "InflaBERT · inflación", result["inflabert"],
                        {"up": "p_up", "neu": "p_neu", "down": "p_down"})
    show_classification(finbert_col, "FinBERT · sentimiento de mercado", result["finbert"],
                        {"pos": "p_pos", "neu": "p_neu", "neg": "p_neg"})


st.title("¿Pueden las noticias anticipar la inflación?")
with st.sidebar:
    model = st.radio("Modelo", list(MODELS), format_func=MODELS.get)
    st.caption(f"API: {API_URL}")

nowcast_view, classifier_view = st.tabs(["Nowcast del IPC", "Clasificador de titulares"])
try:
    with nowcast_view:
        nowcast_tab(model)
    with classifier_view:
        classifier_tab()
except requests.ConnectionError:
    st.error(f"No se puede conectar con la API en {API_URL}. Arráncala con `make run_api`.")
except RuntimeError as exc:
    st.error(f"La API devolvió un error: {exc}")
