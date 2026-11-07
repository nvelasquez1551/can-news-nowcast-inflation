"""Datos: descarga FRED, carga del CSV raw y del índice de noticias, split por fecha.

Copia la lógica del notebook 01 (descarga) y de los pasos 1 y 9 del notebook 02
(carga y split). Este módulo NO transforma nada: eso vive en features.py.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from fredapi import Fred

from nowcast import params


def download_fred(out_path: Path = params.RAW_MACRO_PATH,
                  overwrite: bool = False) -> pd.DataFrame:
    """Descarga las series FRED, las lleva a frecuencia mensual y guarda el CSV raw.

    Series mensuales: media mensual sobre índice de inicio de mes (MS).
    Series diarias/semanales (oil, gas): columnas `<name>_mean` y `<name>_last`.
    """
    out_path = Path(out_path)
    # FRED revisa datos históricos: re-descargar puede cambiar los números de v1.
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"{out_path} already exists; pass overwrite=True to replace it")
    if params.FRED_API_KEY is None:
        raise RuntimeError("FRED_API_KEY is not set (put it in .env)")

    fred = Fred(api_key=params.FRED_API_KEY)
    monthly = {}
    for name, series_id in params.FRED_SERIES.items():
        series = fred.get_series(series_id,
                                 observation_start=params.FRED_START,
                                 observation_end=params.FRED_END)
        resampled = series.resample("MS")
        if name in params.HIGH_FREQ_SERIES:
            monthly[f"{name}_mean"] = resampled.mean()
            monthly[f"{name}_last"] = resampled.last()
        else:
            monthly[name] = resampled.mean()

    df = pd.DataFrame(monthly)
    df.index.name = params.RAW_DATE_COL

    # Chequeo del índice (notebook 01, paso 7)
    expected = pd.date_range(params.FRED_START, params.FRED_END, freq="MS")
    assert df.index.is_unique, "duplicate months in index"
    assert df.index.equals(expected), "index does not cover FRED_START..FRED_END"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    print(f"Saved {df.shape} to {out_path}")
    return df


def load_raw_data(path: Path = params.RAW_MACRO_PATH) -> pd.DataFrame:
    """Carga el CSV raw mensual (salida de download_fred / notebook 01)."""
    raw = pd.read_csv(path, index_col=params.RAW_DATE_COL, parse_dates=True)
    raw.index.freq = "MS"

    expected_cols = ({"cpi"} | set(params.LOG_DIFF_COLS)
                     | set(params.LEVEL_COLS))
    missing_cols = expected_cols - set(raw.columns)
    assert not missing_cols, f"columns missing from raw data: {missing_cols}"
    return raw


def load_news_index(path: Path = params.NEWS_INDEX_PATH) -> pd.DataFrame | None:
    """Carga el índice mensual de noticias del Modelo 1.

    Devuelve None si el archivo no existe (todo corre solo con datos macro).
    El índice del DataFrame es el mes (inicio de mes), igual que el raw macro.
    """
    path = Path(path)
    if not path.exists():
        return None

    news = pd.read_csv(path, parse_dates=[params.NEWS_DATE_COL])
    missing_cols = set(params.NEWS_COLS) - set(news.columns)
    assert not missing_cols, f"columns missing from news index: {missing_cols}"

    news = news.set_index(params.NEWS_DATE_COL)[params.NEWS_COLS]
    news.index = news.index.to_period("M").to_timestamp()  # inicio de mes
    news.index.name = params.RAW_DATE_COL
    return news


def split_by_date(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split temporal sin mezclar: train hasta TRAIN_END, test después (notebook 02, paso 9)."""
    df = df.loc[params.TRAIN_START:params.TEST_END].copy()
    df["split"] = np.where(df.index <= params.TRAIN_END, "train", "test")

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    assert len(test) == params.N_TEST_MONTHS, (
        f"expected {params.N_TEST_MONTHS} test months, got {len(test)}")
    return train, test
