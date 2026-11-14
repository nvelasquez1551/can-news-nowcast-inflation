"""Fixtures sintéticas para los tests del paquete nowcast.

Ningún test usa la API de FRED, los CSV de data/ ni los meses reales de test.
"""

import numpy as np
import pandas as pd
import pytest

from nowcast import params
from nowcast.ml_logic.features import build_model_table

# Grilla mínima de XGBoost para que los tests corran rápido
SMALL_GRID = {
    "model__n_estimators": [5, 10],
    "model__max_depth": [1],
    "model__learning_rate": [0.1],
}


def make_raw(seed: int = 0) -> pd.DataFrame:
    """Tabla raw mensual sintética con las mismas columnas que el CSV de FRED."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(params.FRED_START, params.FRED_END, freq="MS",
                        name=params.RAW_DATE_COL)
    n = len(idx)

    def level(start, drift, vol):
        return start * np.exp(np.cumsum(drift + vol * rng.normal(size=n)))

    gas_mean = level(1.1, 0.002, 0.05)
    oil_mean = level(17.0, 0.003, 0.08)
    raw = pd.DataFrame({
        "cpi": level(162.0, 0.002, 0.002),
        "cpi_core": level(171.6, 0.002, 0.001),
        "oil_mean": oil_mean,
        "oil_last": oil_mean * np.exp(0.02 * rng.normal(size=n)),
        "gas_mean": gas_mean,
        "gas_last": gas_mean * np.exp(0.02 * rng.normal(size=n)),
        "exp": 3 + 0.5 * rng.normal(size=n),
        "ppi": level(125.0, 0.002, 0.01),
        "housing": level(1500.0, 0.0, 0.08),
        "payems": level(125000.0, 0.001, 0.002),
    }, index=idx)
    raw.index.freq = "MS"
    return raw


@pytest.fixture
def raw() -> pd.DataFrame:
    return make_raw()


@pytest.fixture
def table(raw) -> pd.DataFrame:
    return build_model_table(raw)
