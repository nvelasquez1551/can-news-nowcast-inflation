"""Streamlit demo: nowcast vs. official CPI, and the headline classifier.

Run from the repo root:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PREDICTIONS_PATH = REPO_ROOT / "data" / "processed" / "model2_predictions.csv"
EXAMPLES = [
    "Gasoline prices soar as crude rallies",
    "Consumer prices fall for a second month",
    "Fed holds rates steady",
]
ROLE_TEXT = {
    "up": "Sube",
    "down": "Baja",
    "neu": "Neutral",
    "pos": "Positivo",
    "neg": "Negativo",
}

st.set_page_config(page_title="Nowcast de inflación", layout="wide")
st.title("¿Pueden las noticias anticipar la inflación?")

nowcast_tab, classifier_tab = st.tabs(["Nowcast vs. dato oficial", "Clasificador de titulares"])


@st.cache_data
def load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["month"])
    return frame.set_index("month").sort_index()


with nowcast_tab:
    if not PREDICTIONS_PATH.exists():
        st.info(
            "Todavía no hay predicciones del Modelo 2. "
            f"Esta vista lee `{PREDICTIONS_PATH.relative_to(REPO_ROOT)}` "
            "(columnas `month`, `y` y una por modelo)."
        )
    else:
        predictions = load_predictions(PREDICTIONS_PATH)
        model_cols = [col for col in predictions.columns if col != "y"]
        chosen = st.multiselect("Modelos", model_cols, default=model_cols[:2])
        st.line_chart(predictions[["y", *chosen]], y_label="IPC, variación mensual (%)")

        from src.model2.evaluate import compare_forecasts

        if chosen:
            baseline = st.selectbox("Comparar contra", model_cols)
            table = compare_forecasts(
                predictions, target="y", forecasts=chosen, baseline=baseline
            )
            st.dataframe(table, hide_index=True)


@st.cache_resource(show_spinner="Cargando FinBERT e InflaBERT...")
def load_classifier():
    from src.model1 import predict

    predict.classify_headline(EXAMPLES[0])
    return predict.classify_headline


def show_result(column, title: str, result: dict, prob_keys: dict[str, str]) -> None:
    with column:
        st.subheader(title)
        st.metric("Etiqueta", ROLE_TEXT.get(result["role"], result["label"]))
        st.metric("Puntaje (-1 a 1)", f"{result['score']:+.2f}")
        probs = pd.Series({ROLE_TEXT[role]: result[key] for role, key in prob_keys.items()})
        st.bar_chart(probs, y_label="Probabilidad")


with classifier_tab:
    st.caption(
        "InflaBERT mide si el titular implica que los precios suben o bajan. "
        "FinBERT mide si es una noticia buena o mala para los mercados."
    )
    example = st.selectbox("Ejemplo", ["", *EXAMPLES])
    headline = st.text_input("Titular (en inglés)", value=example)
    if headline.strip():
        classify_headline = load_classifier()
        result = classify_headline(headline)
        infla_col, finbert_col = st.columns(2)
        show_result(
            infla_col,
            "InflaBERT · inflación",
            result["inflabert"],
            {"up": "p_up", "neu": "p_neu", "down": "p_down"},
        )
        show_result(
            finbert_col,
            "FinBERT · sentimiento de mercado",
            result["finbert"],
            {"pos": "p_pos", "neu": "p_neu", "neg": "p_neg"},
        )
