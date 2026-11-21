"""Frontend del nowcast. Solo llama a la API (nowcast/api/fast.py) y muestra los resultados.

    API_URL=http://localhost:8000 streamlit run app/streamlit_app.py

La dirección de la API sale de la variable de entorno API_URL o, si no está, de
st.secrets["API_URL"] (Streamlit Cloud: Settings > Secrets). Sin ninguna de las dos,
la app muestra un aviso y se detiene.
"""

from __future__ import annotations

import os

import altair as alt
import pandas as pd
import requests
import streamlit as st


def read_api_url() -> str:
    """API_URL del entorno; si no está, la de st.secrets. Vacía si no hay ninguna."""
    url = os.environ.get("API_URL", "").strip()
    if not url:
        try:
            url = str(st.secrets.get("API_URL", "")).strip()
        except Exception:  # sin secrets.toml, Streamlit lanza un error al leer st.secrets
            url = ""
    return url.rstrip("/")


API_URL = read_api_url()

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


def call(method: str, path: str, **kwargs):
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


SOURCES = {"fnspid": "FNSPID", "alpaca": "Alpaca/Benzinga"}


def source_name(source: str | None) -> str:
    return SOURCES.get(source, source or "fuente desconocida")


def headline_line(item: dict) -> str:
    return (f"- {item['title']}  \n  <small>{item['date']} · {source_name(item['source'])} · "
            f"puntaje {num(item['score'], sign=True)}</small>")


def headlines_off(response: dict) -> bool:
    """La API con HEADLINES_ENABLED=0 responde enabled=false y un aviso."""
    if response.get("enabled", True):
        return False
    st.info(f"Los titulares no están disponibles en este despliegue "
            f"({response.get('message', '')}).")
    return True


def news_section(month: str, result: dict, drivers: pd.DataFrame) -> None:
    st.markdown("### Las noticias del mes")
    news = api_get("/headlines", month=month)
    if headlines_off(news):
        return
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

    sources = ", ".join(source_name(x) for x in news.get("sources", []))
    if sources:
        st.caption(f"Fuente de los titulares de este mes: {sources}. Puntaje de InflaBERT "
                   "por titular, de -1 (menos inflación) a +1 (más inflación).")

    left, right = st.columns(2)
    for column, key, title in ((left, "up", "Los que más empujaron al alza"),
                               (right, "down", "Los que más empujaron a la baja")):
        with column:
            st.markdown(f"**{title}**")
            for item in news[key]:
                st.markdown(headline_line(item), unsafe_allow_html=True)

    query = st.text_input(f"Buscar en los titulares de {month_name(month)}",
                          placeholder="oil, Fed, tariffs...", key=f"month_search_{month}").strip()
    if len(query) >= 2:
        found = api_get("/headlines/search", q=query, k=10, month=month)
        if not found["results"]:
            st.caption(f"Ningún titular de {month_name(month)} contiene «{query}».")
        else:
            st.caption(f"{found['n_matches']:,} titulares de {month_name(month)} contienen "
                       f"«{query}». Los más recientes:".replace(",", "."))
            for item in found["results"]:
                st.markdown(headline_line(item), unsafe_allow_html=True)


def get_comparison() -> dict | None:
    try:
        return api_get("/comparison")
    except RuntimeError:
        return None


def comparison_section(result: dict | None) -> None:
    """Con y sin noticias, fuera de muestra: re-ajustadas con datos hasta 2020-12."""
    st.markdown("### ¿Ayudan las noticias a estimar mejor?")
    if result is None:
        st.caption("La comparación necesita el índice de noticias y los datos de prueba completos.")
        return
    without, with_news = result["without"]["rmse"], result["with"]["rmse"]
    left, right = st.columns(2)
    left.metric("Error sin noticias (RMSE)", num(without, 3))
    right.metric("Error con noticias (RMSE)", num(with_news, 3),
                 delta=num(with_news - without, 3, sign=True), delta_color="inverse")
    if result["significant_5pct"]:
        better = "más bajo" if with_news < without else "más alto"
        verdict = f"Con noticias el error es {better} y la diferencia es significativa"
    else:
        verdict = "La diferencia **no es significativa**: las noticias no mejoran la estimación"
    st.write(f"{verdict} (prueba de Diebold-Mariano, p = {num(result['p_value'], 2)}).")
    st.caption(f"Comparación fuera de muestra: los dos modelos lineales se entrenaron con datos "
               f"de {result['train_start']} a {result['train_end']} y se evaluaron en "
               f"{result['n_months']} meses que no vieron ({result['test_start']}.."
               f"{result['test_end']}; sin {', '.join(result['excluded_months'])}, que no "
               "tienen dato oficial de IPC). RMSE en puntos de inflación mensual.")


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
    if headlines_off(keywords):
        return
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
        results = {f"{r['date']} · {r['title']} ({source_name(r['source'])})": r
                   for r in found["results"]}
        st.caption(f"{found['n_matches']:,} titulares contienen «{query}». "
                   "Estos son los más recientes:".replace(",", "."))
        choice = st.radio("Sugerencias", list(results), index=None, label_visibility="collapsed")
        if choice:
            item = results[choice]
            show_verdict(item["role"], item["score"], month_name(item["date"][:7]))

    st.caption("Revisión manual de 50 titulares del índice elegidos al azar, con un solo "
               "etiquetador y una regla escrita antes de etiquetar: InflaBERT acierta el 32 % "
               "(IC 95 % 21-46 %), mientras que decir siempre «neutral» acierta el 42 %; kappa "
               "0,02. En los 29 titulares con dirección de precios coincide en la dirección el "
               "55 % (IC 95 % 38-72 %). El 42 % de los titulares (21 de 50) no tenía dirección "
               "de precios, y el modelo casi nunca predice «neutral». Con 50 titulares y un solo"
               " etiquetador los intervalos son amplios; el resultado no demuestra que el modelo"
               " sea peor que el azar, solo que no se observa un acuerdo claro.")


ROLE_ES = {"up": "más inflación ↑", "down": "menos inflación ↓", "neu": "neutral (nada claro)"}


def try_headline() -> None:
    """Clasifica un titular nuevo con InflaBERT (POST /classify)."""
    with st.container(border=True):
        st.markdown("#### Prueba un titular")
        text = st.text_input("Escribe un titular en inglés", key="try_headline",
                             placeholder="Gas prices jump as refineries shut down").strip()
        if not st.button("Clasificar", disabled=not text):
            st.caption("El modelo leerá el titular y dirá si apunta a más o menos inflación.")
            return
        with st.spinner("Cargando el modelo y leyendo el titular (la primera vez tarda)..."):
            try:
                response = call("POST", "/classify", json={"headline": text})
            except requests.RequestException as exc:
                st.error(f"No se pudo contactar la API: {exc}")
                return
        body = response.json() if response.headers.get("content-type", "").startswith(
            "application/json") else {}
        if response.status_code == 200 and body.get("enabled") is False:
            st.info("La clasificación de titulares no está disponible en este despliegue "
                    f"({body.get('message', '')}).")
            return
        if response.status_code != 200:
            st.error("No se pudo cargar el modelo de lenguaje (InflaBERT), así que no se puede "
                     f"clasificar el titular. Detalle: {body.get('detail', response.text)}")
            return
        infla = body["inflabert"]
        left, right = st.columns(2)
        left.metric("InflaBERT dice", ROLE_ES.get(infla["role"], infla["role"]))
        right.metric("Puntaje", num(infla["score"], 3, sign=True))
        st.caption(f"Probabilidades: más {infla['p_up']:.0%} · menos {infla['p_down']:.0%} · "
                   f"neutral {infla['p_neu']:.0%}.")
        st.caption("Ojo: el puntaje es la salida raw del modelo (de -1 a +1, sin reescalar ni "
                   "promediar), y InflaBERT se entrenó con otro tipo de noticias, así que "
                   "en estos titulares puede equivocarse.")


def how_to_use(status: dict) -> None:
    with st.container(border=True):
        st.markdown("#### Cómo usar")
        st.markdown(
            "**Eliges** un mes.  \n"
            "**Recibes**: la estimación (nowcast) de la inflación mensual del IPC de ese mes, "
            "en %; el dato oficial, cuando existe; los factores que movieron la estimación; "
            "y la comparación del modelo con y sin noticias.  \n"
            "**Lo que no hace**: no pronostica el futuro, solo estima meses que ya "
            f"terminaron. Y el modelo guardado se entrenó con los meses de "
            f"{status['train_start']} a {status['train_end']} (incluido), así que esos meses "
            "ya los conocía: su estimación no es fuera de muestra.")


st.markdown('<div class="eyebrow">NOWCAST DE INFLACIÓN · EE. UU.</div>', unsafe_allow_html=True)
st.title("¿Cuánto subieron los precios este mes?")
st.markdown('<p class="lead">El dato oficial de inflación sale unas dos semanas después de '
            'terminar el mes. Este modelo lo estima antes, con el petróleo, la gasolina, '
            'la inflación de meses anteriores y las noticias.</p>', unsafe_allow_html=True)

if not API_URL:
    st.error("Falta la dirección de la API. Define la variable de entorno `API_URL` o "
             "añade `API_URL = \"https://...\"` en los secrets de Streamlit "
             "(`.streamlit/secrets.toml` o Settings > Secrets en Streamlit Cloud).")
    st.stop()

try:
    how_to_use(api_get("/"))
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

    if result.get("fallback"):
        st.info(f"Se usó el modelo sin noticias: {result['fallback']}.")
    comparison = get_comparison()
    comparison_section(comparison)
    trained = f"Modelo entrenado con datos hasta {result['train_end']}"
    if comparison is not None:
        trained += (f"; la comparación es fuera de muestra (entrenada hasta "
                    f"{comparison['train_end']}, evaluada {comparison['test_start']}.."
                    f"{comparison['test_end']})")
    st.caption(trained + ".")

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
    if result["sample"] == "in_sample":
        st.caption(f"Ojo: el modelo se entrenó con datos hasta {result['train_end']}, así que "
                   "este mes ya lo conocía. Esta estimación no es un pronóstico fuera de muestra.")
    elif result["sample"] == "before_training":
        st.caption(f"Este mes es anterior a los datos de entrenamiento "
                   f"(desde {result['train_start']}).")
    else:
        st.caption("El modelo no había visto este mes al entrenarse.")

    news_section(month, result, drivers)

    headline_search()
    try_headline()
except requests.ConnectionError:
    st.error(f"No hay conexión con la API ({API_URL}). Arráncala con `make run_api`.")
except RuntimeError as exc:
    st.error(f"La API respondió con un error: {exc}")
