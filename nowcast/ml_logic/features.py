"""Features: target y y las features del notebook 02, más `month` (notebook 03 v1).

Son funciones fijas: no aprenden nada del train (sin medias, sin escalado).
Todo lo que se ajusta con datos de train vive en preprocessor.py.
"""

import numpy as np
import pandas as pd

from nowcast import params


def source_of(col: str) -> str:
    """Serie de origen de una columna (p. ej. 'oil_mean_chg' -> 'oil')."""
    col = col.removesuffix("_chg")
    return col.split("_")[0] if col.startswith(("oil", "gas")) else col


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Target y las 14 features del notebook 02. No se elimina ninguna fila aquí."""
    out = pd.DataFrame(index=raw.index)

    # Target
    out["y"] = 100 * np.log(raw["cpi"]).diff()
    out["y_annualized"] = 100 * ((1 + out["y"] / 100) ** 12 - 1)

    # Paso 2: cambios mensuales (exp queda en niveles)
    changes = pd.DataFrame(index=raw.index)
    for col in params.LOG_DIFF_COLS:
        changes[f"{col}_chg"] = 100 * np.log(raw[col]).diff()
    for col in params.LEVEL_COLS:
        changes[col] = raw[col]

    # Paso 3: rezagos del IPC
    for k in range(1, params.N_CPI_LAGS + 1):
        out[f"y_lag{k}"] = out["y"].shift(k)

    # Paso 4: medias móviles de valores PASADOS (shift(1) deja fuera el mes t)
    window = params.MA_WINDOW
    out[f"y_ma{window}"] = out["y"].shift(1).rolling(window).mean()
    out[f"cpi_core_chg_ma{window}"] = (
        changes["cpi_core_chg"].shift(1).rolling(window).mean()
    )

    # Paso 5: rezago de disponibilidad, valor en t = valor en t - lag
    for col in changes.columns:
        out[col] = changes[col].shift(params.AVAILABILITY_LAG[source_of(col)])

    return out


def describe_features() -> pd.DataFrame:
    """Tabla de features del notebook 02 (feature_table.csv)."""
    window = params.MA_WINDOW
    rows = []
    for col in params.ALL_FEATURES:
        if col.startswith("y_lag"):
            src, transformation, lag = (
                "cpi", f"y shifted by {col[5:]} month(s)", int(col[5:]))
        elif col.startswith("y_ma"):
            src, transformation, lag = (
                "cpi", f"mean of y over previous {window} months (shift(1))", 1)
        elif col.startswith("cpi_core_chg_ma"):
            src, transformation, lag = (
                "cpi_core",
                f"mean of 100*diff(log) over previous {window} months (shift(1))",
                1)
        else:
            src = source_of(col)
            lag = params.AVAILABILITY_LAG[src]
            if col in params.LEVEL_COLS:
                transformation = f"level, shifted by {lag} month(s)"
            else:
                transformation = (f"100*diff(log({col.removesuffix('_chg')})), "
                                  f"shifted by {lag} month(s)")
        rows.append({"feature": col, "source_series": params.FRED_SERIES[src],
                     "transformation": transformation, "availability_lag": lag})
    return pd.DataFrame(rows)


def build_model_table(raw: pd.DataFrame,
                      news: pd.DataFrame | None = None,
                      use_news: bool = params.USE_NEWS) -> pd.DataFrame:
    """Tabla del modelo: una fila por mes, target + features, sin NaN.

    1. build_features y eliminar filas con NaN en alguna feature o en y (notebook 02).
    2. Quitar DROP_FEATURES y agregar `month` si USE_MONTH (notebook 03 v1).
    3. Solo si use_news=True y hay índice: unir noticias, eliminar meses sin
       noticias y empezar la muestra en el primer mes con noticias.
    """
    features_all = build_features(raw)
    table = features_all.dropna(subset=params.ALL_FEATURES + [params.TARGET])
    table = table.drop(columns=params.DROP_FEATURES)

    if params.USE_MONTH:
        table[params.MONTH_COL] = table.index.month

    if use_news:
        if news is None:
            print("use_news=True but no news index found: running macro-only")
        else:
            table = table.join(news[params.NEWS_COLS], how="left")
            table = table.dropna(subset=params.NEWS_COLS)

    return table


def get_feature_columns(use_news: bool = params.USE_NEWS) -> list[str]:
    """Columnas que entran al modelo: FEATURES macro (+ noticias si use_news)."""
    return params.FEATURES + (params.NEWS_COLS if use_news else [])
