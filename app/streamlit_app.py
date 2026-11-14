"""Frontend del nowcast. Solo llama a la API (nowcast/api/fast.py) y muestra los resultados.

    API_URL=http://localhost:8000 streamlit run app/streamlit_app.py   # con la API aparte
    streamlit run app/streamlit_app.py                                 # API dentro de la app

Sin API_URL (como en Streamlit Cloud) la API corre dentro de la app con los datos de
demo_data/, y los modelos se entrenan al arrancar.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
API_URL = os.environ.get("API_URL", "").rstrip("/")
EMBEDDED = not API_URL
if EMBEDDED:
    # Antes de importar nowcast: params lee estas rutas al importarse
    os.environ.setdefault("NOWCAST_DATA_DIR", str(REPO_ROOT / "demo_data"))
    os.environ.setdefault("NOWCAST_MODELS_DIR", str(Path(tempfile.gettempdir()) / "nowcast_models"))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

NAVY, AMBER, GREY = "#14273d", "#d9962b", "#8a94a3"
MONTHS = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
          "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
FACTOR_NAMES = {
    "Inflación pasada": "Inflación de meses anteriores",
    "Inflación subyacente": "Inflación sin alimentos ni energía",
    "Estacionalidad": "Época del año",
    "Expectativas de inflación": "Lo que espera la gente",
    "Precios al productor": "Precios de las fábricas",
}
NEWS_GROUP = "Noticias"
KEYWORDS_ES = {
    "oil": "petróleo", "prices": "precios", "fed": "Fed", "crude": "crudo",
    "inflation": "inflación", "gasoline": "gasolina", "supply chain": "cadena de suministro",
    "rate hike": "subida de tasas", "cpi": "IPC", "interest rate": "tasa de interés",
    "tariffs": "aranceles", "gas prices": "precio del gas", "federal reserve": "Reserva Federal",
    "deflation": "deflación", "wages": "salarios", "consumer prices": "precios al consumidor",
    "food prices": "precio de los alimentos", "ppi": "precios al productor",
    "energy prices": "precio de la energía", "gasoline prices": "precio de la gasolina",
    "wage growth": "subida de salarios", "core cpi": "IPC subyacente",
    "consumer price index": "índice de precios", "cost of living": "costo de vida",
    "disinflation": "desinflación",
}

st.set_page_config(page_title="Inflación EE. UU.", layout="centered")
st.markdown(f"""
<style>
  .block-container {{ padding-top: 2.5rem; max-width: 860px; }}
  .eyebrow {{ color: {AMBER}; font-weight: 700; letter-spacing: .08em; font-size: .8rem; }}
  .lead {{ color: #4a5768; font-size: 1.05rem; margin-top: -.6rem; }}
  .card {{ background: {NAVY}; color: #f5f3ee; border-radius: 10px; padding: 1.4rem 1.6rem; }}
  .card .label {{ color: {AMBER}; font-size: .8rem; font-weight: 700; letter-spacing: .06em; }}
  .card .big {{ font-size: 2.6rem; font-weight: 700; line-height: 1.1; margin: .3rem 0; }}
  .card .row {{ display: flex; gap: 2.5rem; margin-top: .8rem; flex-wrap: wrap; }}
  .card .row div span {{ display: block; color: #b9c2cf; font-size: .8rem; }}
  .card .row div b {{ font-size: 1.25rem; }}
  .card p {{ color: #d5dbe3; margin: .9rem 0 0; }}
  h3 {{ margin-top: 2rem; }}
</style>
""", unsafe_allow_html=True)


def num(value: float, decimals: int = 2, sign: bool = False) -> str:
    text = f"{value:+.{decimals}f}" if sign else f"{value:.{decimals}f}"
    return text.replace(".", ",")


def month_name(month: str) -> str:
    stamp = pd.Timestamp(month)
    return f"{MONTHS[stamp.month - 1]} {stamp.year}"


@st.cache_resource(show_spinner="Preparando los modelos (solo la primera vez, ~1 minuto)...")
def embedded_client():
    from fastapi.testclient import TestClient

    from nowcast import params
    from nowcast.api.fast import app, available_models
    from nowcast.interface.main import train

    if params.PRIMARY_MODEL not in available_models():
        train()
    if f"{params.PRIMARY_MODEL}{params.NEWS_MODEL_SUFFIX}" not in available_models():
        train(use_news=True)
    return TestClient(app)


def call(method: str, path: str, **kwargs):
    if EMBEDDED:
        return embedded_client().request(method, path, **kwargs)
    return requests.request(method, f"{API_URL}{path}", timeout=300, **kwargs)


@st.cache_data(ttl=60, show_spinner=False)
def api_get(path: str, **query) -> dict:
    response = call("GET", path, params=query)
    if response.status_code != 200:
        raise RuntimeError(response.json().get("detail", response.text))
    return response.json()


def explain(result: dict, drivers: pd.DataFrame) -> str:
    previous = month_name(result["previous_month"]).split()[0]
    change = result["change"]
    if abs(change) < 0.01:
        text = f"La estimación quedó casi igual que en {previous}."
    else:
        verb = "subió" if change > 0 else "bajó"
        text = f"Frente a {previous}, la estimación {verb} {num(abs(change))} puntos."
    same_way = drivers[drivers["change"] * (1 if change >= 0 else -1) > 0.005]
    if not same_way.empty:
        main = same_way.iloc[0]["factor"].lower()
        direction = "hacia arriba" if change >= 0 else "hacia abajo"
        text += f" Lo que más la empujó {direction} fue: {main}."
    return text


def factors_chart(drivers: pd.DataFrame) -> alt.Chart:
    data = drivers[drivers["change"].abs() >= 0.005].head(5)
    if NEWS_GROUP in drivers["group"].values and NEWS_GROUP not in data["group"].values:
        data = pd.concat([data, drivers[drivers["group"] == NEWS_GROUP]])
    data = data.copy()
    data["label"] = data["change"].map(lambda v: num(v, sign=True))
    data["sentido"] = data["change"].ge(0).map({True: "sube", False: "baja"})
    bars = alt.Chart(data).mark_bar(cornerRadius=3, height=22).encode(
        x=alt.X("change:Q", title=None, scale=alt.Scale(padding=50),
                axis=alt.Axis(labels=False, ticks=False, grid=False)),
        y=alt.Y("factor:N", sort=None, title=None,
                axis=alt.Axis(labelLimit=260, labelFontSize=13, ticks=False, domain=False)),
        color=alt.Color("sentido:N", legend=None,
                        scale=alt.Scale(domain=["sube", "baja"], range=[AMBER, NAVY])),
        tooltip=[alt.Tooltip("factor:N", title="Factor"), alt.Tooltip("label:N", title="Puntos")],
    )
    # El número va fuera de la barra: a la derecha si suma, a la izquierda si resta
    right = bars.mark_text(align="left", dx=6, fontSize=12).encode(
        text="label:N", color=alt.value(NAVY)).transform_filter("datum.change >= 0")
    left = bars.mark_text(align="right", dx=-6, fontSize=12).encode(
        text="label:N", color=alt.value(NAVY)).transform_filter("datum.change < 0")
    return (bars + right + left).properties(height=alt.Step(36)).configure_view(stroke=None)


def history_chart(history: pd.DataFrame, month: str) -> alt.Chart:
    data = history.melt(id_vars="month", value_vars=["actual", "nowcast"],
                        var_name="serie", value_name="valor")
    data["serie"] = data["serie"].map({"actual": "Dato oficial", "nowcast": "Estimación"})
    lines = alt.Chart(data).mark_line(strokeWidth=2).encode(
        x=alt.X("month:T", title=None, axis=alt.Axis(format="%Y", tickCount="year", grid=False)),
        y=alt.Y("valor:Q", title="% en el mes"),
        color=alt.Color("serie:N", title=None, legend=alt.Legend(orient="top"),
                        scale=alt.Scale(domain=["Estimación", "Dato oficial"],
                                        range=[AMBER, NAVY])),
        tooltip=[alt.Tooltip("month:T", format="%m/%Y", title="Mes"),
                 alt.Tooltip("serie:N", title=" "),
                 alt.Tooltip("valor:Q", format=".2f", title="%")],
    )
    marker = alt.Chart(pd.DataFrame({"month": [pd.Timestamp(month)]})).mark_rule(
        color=GREY, strokeDash=[3, 3]).encode(x="month:T")
    return (lines + marker).properties(height=280).configure_view(stroke=None)


def news_section(month: str, result: dict, drivers: pd.DataFrame) -> None:
    st.markdown("### Las noticias del mes")
    news = api_get("/headlines", month=month)
    if not news["available"]:
        st.caption("Aparecerán aquí cuando esté listo el índice de noticias (Modelo 1).")
        return
    st.write(f"En {month_name(month)} hubo {news['n_articles']:,} noticias sobre precios. "
             f"El {news['share_up']:.0%} apuntaba a más inflación y el "
             f"{news['share_down']:.0%} a menos.".replace(",", "."))

    if result["news"] is None:
        st.caption("El modelo elegido no usa noticias. Elige uno «con noticias» en Ajustes "
                   "para ver cuánto movieron la estimación.")
    else:
        score = result["news"]["infla_score_mean"]
        previous = month_name(result["previous_month"]).split()[0]
        tone = "más" if score["value"] > score["previous"] else "menos"
        effect = drivers.loc[drivers["group"] == NEWS_GROUP, "change"].iloc[0]
        st.write(f"Frente a {previous}, las noticias fueron {tone} inflacionarias "
                 f"(puntaje medio {num(score['previous'], sign=True)} → "
                 f"{num(score['value'], sign=True)}, de -1 a +1). "
                 f"Eso movió la estimación {num(effect, sign=True)} puntos.")

    left, right = st.columns(2)
    for column, key, title in ((left, "up", "Apuntan a más inflación"),
                               (right, "down", "Apuntan a menos inflación")):
        with column:
            st.markdown(f"**{title}**")
            for item in news[key]:
                st.markdown(f"- {item['title']}")


def comparison_section(model: str) -> None:
    st.markdown("### ¿Ayudan las noticias a estimar mejor?")
    try:
        result = api_get("/comparison", model=model)
    except RuntimeError:
        st.caption("Para comparar hace falta entrenar el modelo con y sin noticias "
                   "(`make train` y `make train_news`).")
        return
    without, with_news = result["without"]["mae"], result["with"]["mae"]
    period = f"{month_name(result['start'])} y {month_name(result['end'])}"
    if abs(with_news - without) < 0.01:
        verdict = ("Con y sin noticias se equivoca prácticamente lo mismo: las noticias "
                   "**no mejoran** la estimación.")
    elif with_news < without and result["p_value"] < 0.1:
        verdict = "Con estos datos, las noticias **sí mejoran** la estimación."
    elif with_news < without:
        verdict = ("Con noticias se equivocó un poco menos, pero la diferencia es tan pequeña "
                   "que podría ser casualidad.")
    else:
        verdict = "Con estos datos, las noticias **no mejoran** la estimación."
    st.write(f"Entre {period}, meses que el modelo no vio al entrenarse, la estimación se "
             f"equivocó en promedio **{num(without)} puntos** sin noticias y "
             f"**{num(with_news)} puntos** con noticias. {verdict}")
    data = pd.DataFrame({"modelo": ["Sin noticias", "Con noticias"],
                         "error": [without, with_news]})
    data["label"] = data["error"].map(num)
    bars = alt.Chart(data).mark_bar(cornerRadius=3, height=26).encode(
        x=alt.X("error:Q", title=None, axis=alt.Axis(labels=False, ticks=False, grid=False)),
        y=alt.Y("modelo:N", sort=None, title=None,
                axis=alt.Axis(labelFontSize=13, ticks=False, domain=False)),
        color=alt.Color("modelo:N", legend=None,
                        scale=alt.Scale(domain=["Sin noticias", "Con noticias"],
                                        range=[NAVY, AMBER])),
    )
    labels = bars.mark_text(align="left", dx=6, fontSize=12).encode(
        text="label:N", color=alt.value(NAVY))
    st.altair_chart((bars + labels).properties(height=alt.Step(40)).configure_view(stroke=None),
                    width="stretch")
    first_year, last_year = result["train_start"][:4], result["train_end"][:4]
    st.caption(f"Error medio por mes, en puntos de inflación mensual. Para que la comparación "
               f"sea justa, los dos modelos aprendieron con los mismos años ({first_year}–"
               f"{last_year}): antes de {first_year} no hay noticias. Los meses del COVID "
               "(primavera de 2020) son los más difíciles para cualquier modelo.")


VERDICT = {"up": "más inflación", "down": "menos inflación", "neu": "nada claro sobre precios"}


def show_verdict(role: str, score: float, when: str | None = None) -> None:
    source = f"Este titular ({when})" if when else "Este titular"
    st.markdown(f"{source} sugiere **{VERDICT[role]}**, según InflaBERT "
                f"(puntaje {num(score, sign=True)}, de -1 a +1).")


def headline_search() -> None:
    st.markdown("### Busca un titular")
    st.caption("Estas son las palabras clave con las que el modelo eligió las noticias sobre "
               "precios, con cuántos titulares tiene cada una. Elige una o escribe otras "
               "palabras en inglés.")
    keywords = api_get("/headlines/keywords")
    picked = None
    if keywords["available"]:
        counts = {k["keyword"]: k["n_headlines"] for k in keywords["keywords"]}
        picked = st.pills(
            "Palabras clave", list(counts), label_visibility="collapsed",
            format_func=lambda kw: f"{KEYWORDS_ES.get(kw, kw)} · {counts[kw]:,}".replace(",", "."))
    typed = st.text_input("…o escribe otras palabras", placeholder="gasoline, food prices, Fed...")
    query = typed.strip() or picked or ""
    if len(query) < 2:
        return

    found = api_get("/headlines/search", q=query, k=8, keyword=not typed.strip())
    if not found["available"]:
        st.caption("La búsqueda estará disponible cuando esté listo el índice de noticias.")
    elif not found["results"]:
        st.caption(f"Ningún titular contiene «{query}».")
    else:
        results = {f"{r['date']} · {r['title']}": r for r in found["results"]}
        st.caption(f"{found['n_matches']:,} titulares contienen «{query}». "
                   "Estos son los más recientes:".replace(",", "."))
        choice = st.radio("Sugerencias", list(results), index=None, label_visibility="collapsed")
        if choice:
            item = results[choice]
            show_verdict(item["role"], item["score"], month_name(item["date"][:7]))

    if typed.strip() and st.button(f"Analizar «{query}» como titular nuevo"):
        with st.spinner("Leyendo el titular..."):
            response = call("POST", "/classify", json={"headline": query})
        if response.status_code == 200:
            infla = response.json()["inflabert"]
            show_verdict(infla["role"], infla["score"])
        else:
            st.caption("Analizar texto nuevo necesita los modelos de lenguaje, que no están "
                       "instalados aquí. Puedes elegir un titular de las sugerencias.")
    st.caption("InflaBERT acierta más o menos 6 de cada 10 veces, así que algunos "
               "titulares salen mal clasificados.")


st.markdown('<div class="eyebrow">NOWCAST DE INFLACIÓN · EE. UU.</div>', unsafe_allow_html=True)
st.title("¿Cuánto subieron los precios este mes?")
st.markdown('<p class="lead">El dato oficial de inflación sale unas dos semanas después de '
            'terminar el mes. Este modelo lo estima antes, con el petróleo, la gasolina, '
            'la inflación de meses anteriores y las noticias.</p>', unsafe_allow_html=True)

try:
    catalog = api_get("/models")
    options = {m["name"]: f"{m['label']} · {'con' if m['uses_news'] else 'sin'} noticias"
               for m in catalog["models"]}
    if not options:
        st.error("No hay modelos entrenados. Ejecuta `make train` (y `make train_news`).")
        st.stop()
    with st.expander("Ajustes"):
        model = st.radio("Modelo", list(options), format_func=options.get,
                         index=list(options).index(catalog["default"]))
    months = api_get("/months", model=model)["months"]
    month = st.selectbox("Mes", months[::-1], format_func=month_name)

    result = api_get("/nowcast", month=month, model=model)
    drivers = pd.DataFrame(result["drivers"])
    drivers["factor"] = drivers["group"].map(lambda g: FACTOR_NAMES.get(g, g))
    gap = result["nowcast"] - result["actual"]

    st.markdown(f"""
    <div class="card">
      <div class="label">ESTIMACIÓN PARA {month_name(month).upper()}</div>
      <div class="big">{num(result['nowcast'])} %</div>
      <div class="row">
        <div><span>Dato oficial</span><b>{num(result['actual'])} %</b></div>
        <div><span>Diferencia</span><b>{num(abs(gap))} puntos</b></div>
        <div><span>Al año, a este ritmo</span><b>{num(result['nowcast_annualized'], 1)} %</b></div>
      </div>
      <p>Los precios habrían subido {num(result['nowcast'])} % en el mes. El dato oficial fue
      {num(result['actual'])} %.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Qué movió la estimación")
    st.write(explain(result, drivers))
    st.altair_chart(factors_chart(drivers), width="stretch")
    st.caption("Ámbar: empujó la estimación hacia arriba. Azul: hacia abajo. "
               "En puntos de inflación mensual, frente al mes anterior.")

    st.markdown("### Estimación frente al dato oficial")
    history = pd.DataFrame(api_get("/history", model=model)["rows"])
    history["month"] = pd.to_datetime(history["month"])
    start = pd.Timestamp(month) - pd.DateOffset(years=4)
    st.altair_chart(history_chart(history[history["month"].between(start, pd.Timestamp(month))],
                                  month), width="stretch")
    if result["in_sample"]:
        st.caption("Ojo: el modelo aprendió con datos hasta 2019, así que este mes ya lo conocía.")
    else:
        st.caption("Desde 2020 el modelo no había visto estos datos: es una prueba real.")

    news_section(month, result, drivers)
    comparison_section(model)

    headline_search()
except requests.ConnectionError:
    st.error(f"No hay conexión con la API ({API_URL}). Arráncala con `make run_api`.")
except RuntimeError as exc:
    st.error(f"La API respondió con un error: {exc}")
