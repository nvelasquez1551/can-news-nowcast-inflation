"""Parámetros y rutas del nowcast de IPC (Modelo 2).

Todas las decisiones del experimento viven aquí. Ningún otro módulo debe
tener rutas, fechas o hiperparámetros escritos a mano.
Los valores copian los notebooks 01, 02, 03_train_macro_models_v1 y
03b_expanding_refit_v1 (fuente de verdad).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

load_dotenv(ROOT / ".env")
FRED_API_KEY = os.environ.get("FRED_API_KEY")  # nunca se versiona (.env)

RAW_DATA_DIR = ROOT / "data" / "raw"
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"  # registry (joblib + metadata json), ignorado por git
# Archivos del registry: models/<modelo>_<timestamp>.joblib (+ .json)
# Ancho fijo y de mayor a menor unidad: el orden alfabético = orden temporal.
# Microsegundos (%f) para que dos guardados seguidos no choquen.
REGISTRY_TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S-%f"

RAW_MACRO_PATH = RAW_DATA_DIR / "macro_monthly_raw.csv"
NEWS_INDEX_PATH = PROCESSED_DATA_DIR / "news_index_monthly.csv"  # salida del Modelo 1

# Salidas del paquete (nombres nuevos: nunca pisan los archivos v1)
CV_RESULTS_PATH = PROCESSED_DATA_DIR / "nowcast_cv_results.csv"
FIXED_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "nowcast_fixed_predictions.csv"
FIXED_RESULTS_PATH = PROCESSED_DATA_DIR / "nowcast_fixed_results.csv"
FIXED_DM_PATH = PROCESSED_DATA_DIR / "nowcast_fixed_dm_tests.csv"
EXPANDING_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "nowcast_expanding_predictions.csv"
EXPANDING_RESULTS_PATH = PROCESSED_DATA_DIR / "nowcast_expanding_results.csv"
EXPANDING_DM_PATH = PROCESSED_DATA_DIR / "nowcast_expanding_dm_tests.csv"

# Salidas de los notebooks v1, usadas solo para verificar el refactor
NB02_MODEL_TABLE_PATH = PROCESSED_DATA_DIR / "model_table_macro.csv"
NB02_FEATURE_TABLE_PATH = PROCESSED_DATA_DIR / "feature_table.csv"
V1_CV_RESULTS_PATH = PROCESSED_DATA_DIR / "cv_results_macro_v1.csv"
V1_TEST_RESULTS_PATH = PROCESSED_DATA_DIR / "results_macro_v1.csv"
V1_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "model_predictions_macro_v1.csv"
V1_DM_TESTS_PATH = PROCESSED_DATA_DIR / "dm_tests_macro_v1.csv"
V1_BEST_PARAMS_PATH = PROCESSED_DATA_DIR / "best_params_macro_v1.json"

# ---------------------------------------------------------------------------
# Descarga FRED (notebook 01)
# ---------------------------------------------------------------------------
FRED_START, FRED_END = "1998-01-01", "2023-12-31"
FRED_SERIES = {
    "cpi": "CPIAUCSL",
    "cpi_core": "CPILFESL",
    "oil": "DCOILWTICO",  # diaria
    "gas": "GASREGW",  # semanal
    "exp": "MICH",
    "ppi": "PPIACO",
    "housing": "HOUST",
    "payems": "PAYEMS",
}
# Series diarias/semanales: se guardan media mensual y último valor
HIGH_FREQ_SERIES = ["oil", "gas"]
RAW_DATE_COL = "date"

# ---------------------------------------------------------------------------
# Construcción de features (notebook 02)
# ---------------------------------------------------------------------------
# Target: y = 100 * diff(log(cpi)). y_annualized es solo para reportar.
TARGET = "y"
TARGET_COLS = ["y", "y_annualized"]
N_CPI_LAGS = 3
MA_WINDOW = 12

# Meses de retraso antes de poder usar cada serie para el nowcast del mes t
# (al FINAL del mes t). Valores por defecto del notebook 02.
AVAILABILITY_LAG = {
    "cpi_core": 1,
    "ppi": 1,
    "housing": 1,
    "payems": 1,
    "exp": 1,
    "oil": 0,
    "gas": 0,
}
# Columnas raw transformadas a 100 * diff(log(x)); exp (MICH) queda en niveles
LOG_DIFF_COLS = ["cpi_core", "ppi", "housing", "payems",
                 "oil_mean", "oil_last", "gas_mean", "gas_last"]
LEVEL_COLS = ["exp"]

# Las 14 features del notebook 02, en su orden
ALL_FEATURES = [
    "y_lag1", "y_lag2", "y_lag3", "y_ma12", "cpi_core_chg_ma12",
    "cpi_core_chg", "ppi_chg", "housing_chg", "payems_chg",
    "oil_mean_chg", "oil_last_chg", "gas_mean_chg", "gas_last_chg", "exp",
]
DROP_FEATURES = ["oil_last_chg", "gas_last_chg"]

# Mes calendario como feature (v1). Se decide por CV (hasta 2019), nunca por test.
USE_MONTH = True
MONTH_COL = "month"

NUMERIC_FEATURES = [f for f in ALL_FEATURES if f not in DROP_FEATURES]
FEATURES = NUMERIC_FEATURES + ([MONTH_COL] if USE_MONTH else [])

# Fechas para el test de leakage del notebook 02
LEAKAGE_TEST_DATES = ["2005-06-01", "2008-10-01", "2015-01-01",
                      "2020-04-01", "2023-12-01"]

# ---------------------------------------------------------------------------
# Noticias (Modelo 1). Desactivado por defecto: la corrida macro usa toda la muestra.
# ---------------------------------------------------------------------------
USE_NEWS = False
NEWS_DATE_COL = "month"
NEWS_COLS = ["infla_score_mean", "infla_net_share"]

# ---------------------------------------------------------------------------
# Split temporal (notebook 02)
# ---------------------------------------------------------------------------
TRAIN_START, TRAIN_END = "1999-01-01", "2019-12-31"
TEST_START, TEST_END = "2020-01-01", "2023-12-31"
N_TEST_MONTHS = 48

# Subperiodos del test (notebook 03 v1)
TEST_PERIODS = {
    "full": ("2020-01-01", "2023-12-31"),
    "2020_2021": ("2020-01-01", "2021-12-31"),
    "2022_2023": ("2022-01-01", "2023-12-31"),
}

# El test (2020-2023) ya se vio en v0 y v1. Solo se evalúa con RUN_TEST=1.
RUN_TEST = os.environ.get("RUN_TEST", "0").strip().lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Modelos (notebook 03 v1, sin Random Forest)
# ---------------------------------------------------------------------------
SEED = 42
N_SPLITS = 5
SCORING = "neg_root_mean_squared_error"

MODEL_NAMES = ["linear", "xgb"]
PRIMARY_MODEL = "linear"  # comparación principal, declarada de antemano

# Parámetros fijos de XGBoost. n_jobs=1: en un dataset tan chico es más rápido.
XGB_FIXED_PARAMS = {
    "random_state": SEED,
    "objective": "reg:squarederror",
    "n_jobs": 1,
}
# Grilla v1 (los mejores valores quedaron en el borde: lr 0.1, depth 1)
XGB_GRID = {
    "model__n_estimators": [100, 300, 600],
    "model__max_depth": [1, 2, 3],
    "model__learning_rate": [0.01, 0.03, 0.1],
}

# Baselines (notebook 03 v1)
BASELINES = ["naive_lag1", "train_mean"]

# Pares Diebold-Mariano (modelo_a, modelo_b); estadístico negativo = A mejor
DM_PAIRS = [
    ("linear", "naive_lag1"),
    ("xgb", "naive_lag1"),
    ("xgb", "linear"),
]

# ---------------------------------------------------------------------------
# Expanding refit (evaluación principal)
# ---------------------------------------------------------------------------
# Re-búsqueda de hiperparámetros de XGBoost cada RETUNE_EVERY meses de test
# (2020-01, 2021-01, 2022-01, 2023-01). None = parámetros fijos (reproduce 03b).
RETUNE_EVERY = 12

# ---------------------------------------------------------------------------
# Explicación del nowcast (API y frontend)
# ---------------------------------------------------------------------------
# Cada feature se agrupa en un "motivo" legible. La contribución de un grupo es
# la suma de las contribuciones de sus features.
DRIVER_GROUPS = {
    "Inflación pasada": ["y_lag1", "y_lag2", "y_lag3", "y_ma12"],
    "Inflación subyacente": ["cpi_core_chg", "cpi_core_chg_ma12"],
    "Precios al productor": ["ppi_chg"],
    "Petróleo": ["oil_mean_chg", "oil_last_chg"],
    "Gasolina": ["gas_mean_chg", "gas_last_chg"],
    "Expectativas de inflación": ["exp"],
    "Vivienda": ["housing_chg"],
    "Empleo": ["payems_chg"],
    "Estacionalidad": [MONTH_COL],
    "Noticias": NEWS_COLS,
}
# Salida del Modelo 1 por artículo, para mostrar los titulares de cada mes
NEWS_SCORED_PATH = PROCESSED_DATA_DIR / "news_scored.parquet"
N_TOP_HEADLINES = 5
