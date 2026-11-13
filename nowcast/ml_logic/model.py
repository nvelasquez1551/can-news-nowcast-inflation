"""Modelos: regresión lineal (principal) y XGBoost (secundario).

Copia la lógica de los notebooks 03_train_macro_models_v1 y 03b_expanding_refit_v1:
- build_model: Pipeline completo (preprocesador + modelo).
- cross_validate / grid_search: TimeSeriesSplit solo sobre train.
- evaluate: evaluación fija (train 1999-2019, test 2020-2023) con baselines.
- diebold_mariano: pérdida cuadrática, h=1, corrección HLN.
- expanding_forecast: refit mensual, re-búsqueda de XGBoost cada RETUNE_EVERY meses.

evaluate y expanding_forecast miran el test. El freno RUN_TEST vive en
interface/main.py, no aquí: estas funciones son lógica pura.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from nowcast import params
from nowcast.ml_logic.data import split_by_date
from nowcast.ml_logic.preprocessor import build_preprocessor


def build_model(model_name: str,
                feature_columns: list[str] = params.FEATURES,
                model_params: dict | None = None) -> Pipeline:
    """Pipeline sin ajustar. `model_params` usa claves 'model__...' (como la grilla)."""
    if model_name == "linear":
        pipe = Pipeline([
            ("preprocessor", build_preprocessor("linear", feature_columns)),
            ("model", LinearRegression()),
        ])
    elif model_name == "xgb":
        pipe = Pipeline([("model", XGBRegressor(**params.XGB_FIXED_PARAMS))])
    else:
        raise ValueError(f"unknown model: {model_name}")

    if model_params:
        pipe.set_params(**model_params)
    return pipe


def cross_validate(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """RMSE por fold con TimeSeriesSplit(N_SPLITS). Pasar solo datos de train."""
    scores = cross_val_score(pipe, X, y,
                             cv=TimeSeriesSplit(n_splits=params.N_SPLITS),
                             scoring=params.SCORING)
    return -scores


def grid_search(X: pd.DataFrame, y: pd.Series,
                feature_columns: list[str] = params.FEATURES,
                grid: dict = params.XGB_GRID) -> GridSearchCV:
    """GridSearchCV + TimeSeriesSplit para XGBoost (la lineal no tiene hiperparámetros)."""
    search = GridSearchCV(build_model("xgb", feature_columns), grid,
                          cv=TimeSeriesSplit(n_splits=params.N_SPLITS),
                          scoring=params.SCORING)
    search.fit(X, y)
    return search


def search_summary(search: GridSearchCV) -> dict:
    """Mejores parámetros, RMSE medio/std y RMSE por fold del mejor candidato."""
    best = search.best_index_
    return {
        "best_params": search.best_params_,
        "cv_rmse_mean": -search.best_score_,
        "cv_rmse_std": search.cv_results_["std_test_score"][best],
        "cv_rmse_folds": np.array(
            [-search.cv_results_[f"split{k}_test_score"][best]
             for k in range(params.N_SPLITS)]),
    }


def compute_metrics(predictions: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """RMSE y MAE por modelo en el test completo y en los subperiodos."""
    rows = []
    for column in columns:
        row = {"model": column}
        for period, (start, end) in params.TEST_PERIODS.items():
            part = predictions.loc[start:end]
            row[f"rmse_{period}"] = root_mean_squared_error(part["y"], part[column])
            row[f"mae_{period}"] = mean_absolute_error(part["y"], part[column])
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(fitted_models: dict[str, Pipeline],
             train: pd.DataFrame, test: pd.DataFrame,
             feature_columns: list[str] = params.FEATURES
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluación fija (notebook 03 v1, paso 11): modelos ajustados una vez en train.

    Baselines: naive_lag1 = y_lag1; train_mean = media FIJA de y en train.
    Devuelve (predicciones, métricas).
    """
    predictions = pd.DataFrame(index=test.index)
    predictions["y"] = test[params.TARGET]
    predictions["naive_lag1"] = test["y_lag1"]
    predictions["train_mean"] = train[params.TARGET].mean()
    for name, pipe in fitted_models.items():
        predictions[name] = pipe.predict(test[feature_columns])

    results = compute_metrics(predictions, params.BASELINES + list(fitted_models))
    return predictions, results


def diebold_mariano(predictions: pd.DataFrame,
                    pairs: list[tuple[str, str]] = params.DM_PAIRS) -> pd.DataFrame:
    """Test DM con pérdida cuadrática, h=1 y corrección HLN (notebook 03 v1, paso 11b).

    Estadístico negativo = el modelo A tuvo menor pérdida. Para h=1 equivale a un
    t-test pareado sobre los errores al cuadrado: se usa como chequeo.
    """
    rows = []
    for model_a, model_b in pairs:
        error_a = predictions["y"] - predictions[model_a]
        error_b = predictions["y"] - predictions[model_b]

        d = error_a ** 2 - error_b ** 2
        n = len(d)

        dm = d.mean() / np.sqrt(d.var(ddof=0) / n)
        dm_hln = dm * np.sqrt((n - 1) / n)
        p_value = 2 * stats.t.sf(abs(dm_hln), df=n - 1)

        paired_t = stats.ttest_rel(error_a ** 2, error_b ** 2)
        assert np.isclose(dm_hln, paired_t.statistic)
        assert np.isclose(p_value, paired_t.pvalue)

        rows.append({"model_a": model_a, "model_b": model_b,
                     "dm_stat": dm_hln, "p_value": p_value})
    return pd.DataFrame(rows)


def expanding_forecast(model_name: str,
                       table: pd.DataFrame,
                       feature_columns: list[str] = params.FEATURES,
                       retune_every: int | None = params.RETUNE_EVERY,
                       fixed_params: dict | None = None) -> pd.DataFrame:
    """Evaluación principal: para cada mes de test t, ajustar con TODAS las filas
    anteriores a t (datos publicados, no predicciones) y predecir t.

    - El modelo se re-ajusta cada mes.
    - XGBoost: GridSearchCV + TimeSeriesSplit sobre las filas disponibles cada
      `retune_every` meses (2020-01, 2021-01, ...). Con retune_every=None usa
      `fixed_params` todo el periodo (reproduce el notebook 03b).

    Devuelve una fila por mes: month, actual, prediction, baselines,
    último mes de train, hiperparámetros usados y si hubo re-búsqueda.
    """
    if model_name == "xgb" and retune_every is None and fixed_params is None:
        raise ValueError("xgb with retune_every=None needs fixed_params")

    # Mismo split que la evaluación fija (incluye el chequeo de 48 meses de test)
    train, test = split_by_date(table)
    table = pd.concat([train, test])
    test_months = test.index

    current_params = fixed_params if model_name == "xgb" else None
    rows = []
    for i, t in enumerate(test_months):
        history = table[table.index < t]
        X_hist, y_hist = history[feature_columns], history[params.TARGET]

        retuned = (model_name == "xgb" and retune_every is not None
                   and i % retune_every == 0)
        if retuned:
            search = grid_search(X_hist, y_hist, feature_columns)
            current_params = search.best_params_

        pipe = clone(build_model(model_name, feature_columns, current_params))
        pipe.fit(X_hist, y_hist)

        rows.append({
            "month": t,
            "actual": table.loc[t, params.TARGET],
            "prediction": pipe.predict(table.loc[[t], feature_columns])[0],
            "naive_lag1": table.loc[t, "y_lag1"],
            "train_mean": y_hist.mean(),  # media de todas las filas antes de t
            "last_train_month": history.index.max(),
            "n_train": len(history),
            "retuned": retuned,
            "hyperparameters": dict(current_params) if current_params else {},
        })
    return pd.DataFrame(rows)
