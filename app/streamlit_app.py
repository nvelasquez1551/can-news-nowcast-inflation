"""Frontend del nowcast. Solo llama a la API (nowcast/api/fast.py) y muestra los resultados.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

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
MODELS = {"linear": "Regresión lineal", "xgb": "XGBoost"}

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


@st.cache_data(ttl=60, show_spinner=False)
def api_get(path: str, **query) -> dict:
    response = requests.get(f"{API_URL}{path}", params=query, timeout=30)
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
    data = drivers[drivers["change"].abs() >= 0.005].head(5).copy()
    data["label"] = data["change"].map(lambda v: num(v, sign=True))
    data["sentido"] = data["change"].ge(0).map({True: "sube", False: "baja"})
    bars = alt.Chart(data).mark_bar(cornerRadius=3, height=22).encode(
        x=alt.X("change:Q", title=None, axis=alt.Axis(labels=False, ticks=False, grid=False)),
        y=alt.Y("factor:N", sort=None, title=None,
                axis=alt.Axis(labelLimit=260, labelFontSize=13, ticks=False, domain=False)),
        color=alt.Color("sentido:N", legend=None,
                        scale=alt.Scale(domain=["sube", "baja"], range=[AMBER, NAVY])),
        tooltip=[alt.Tooltip("factor:N", title="Factor"), alt.Tooltip("label:N", title="Puntos")],
    )
    labels = bars.mark_text(align="left", dx=6, fontSize=12, color=NAVY).encode(
        text="label:N", color=alt.value(NAVY))
    return (bars + labels).properties(height=alt.Step(36)).configure_view(stroke=None)


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


def headlines(month: str) -> None:
    news = api_get("/headlines", month=month)
    st.markdown("### Noticias del mes")
    if not news["available"]:
        st.caption("Aparecerán aquí cuando esté listo el índice de noticias (Modelo 1).")
        return
    st.caption(f"{news['n_articles']} noticias sobre precios en {month_name(month)}.")
    left, right = st.columns(2)
    for column, key, title in ((left, "up", "Apuntan a más inflación"),
                               (right, "down", "Apuntan a menos inflación")):
        with column:
            st.markdown(f"**{title}**")
            for item in news[key]:
                st.markdown(f"- {item['title']}")


def headline_tester() -> None:
    headline = st.text_input("Escribe un titular en inglés",
                             placeholder="Gasoline prices soar as crude rallies")
    if not headline.strip():
        return
    with st.spinner("Leyendo el titular..."):
        response = requests.post(f"{API_URL}/classify", json={"headline": headline}, timeout=300)
    if response.status_code != 200:
        st.warning(response.json().get("detail", "No se pudo clasificar."))
        return
    infla = response.json()["inflabert"]
    verdict = {"up": "más inflación", "down": "menos inflación", "neu": "nada claro sobre precios"}
    st.markdown(f"Según el modelo, este titular sugiere **{verdict[infla['role']]}** "
                f"(puntaje {num(infla['score'], sign=True)}, de -1 a +1).")


st.markdown('<div class="eyebrow">NOWCAST DE INFLACIÓN · EE. UU.</div>', unsafe_allow_html=True)
st.title("¿Cuánto subieron los precios este mes?")
st.markdown('<p class="lead">El dato oficial de inflación sale unas dos semanas después de '
            'terminar el mes. Este modelo lo estima antes, con el petróleo, la gasolina, '
            'la inflación de meses anteriores y las noticias.</p>', unsafe_allow_html=True)

try:
    with st.expander("Ajustes"):
        model = st.radio("Modelo", list(MODELS), format_func=MODELS.get, horizontal=True)
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

    headlines(month)

    st.markdown("### Prueba con un titular")
    headline_tester()
except requests.ConnectionError:
    st.error(f"No hay conexión con la API ({API_URL}). Arráncala con `make run_api`.")
except RuntimeError as exc:
    st.error(f"La API respondió con un error: {exc}")
