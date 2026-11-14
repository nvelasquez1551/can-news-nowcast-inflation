"""Explicación del nowcast: cuánto aporta cada feature y por qué cambió frente al mes anterior.

- Regresión lineal: contribución = coeficiente x valor transformado (escalado u one-hot).
  La base es el intercepto, es decir, la predicción con todas las features en su media
  de train y en el mes de referencia.
- XGBoost: contribuciones SHAP exactas de XGBoost (`pred_contribs=True`).

En ambos casos base + suma de contribuciones = predicción, así que el cambio entre
dos meses se reparte exactamente entre los grupos de DRIVER_GROUPS.
"""

import numpy as np
import pandas as pd
import xgboost
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from nowcast import params


def feature_contributions(pipe: Pipeline, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Contribución de cada feature original por mes, y el valor base.

    Las columnas one-hot de `month` se suman en una sola columna `month`.
    """
    model = pipe.named_steps["model"]
    columns = list(X.columns)

    if isinstance(model, LinearRegression):
        preprocessor = pipe.named_steps["preprocessor"]
        Z = preprocessor.transform(X)
        Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
        names = preprocessor.get_feature_names_out()
        raw = pd.DataFrame(Z * model.coef_, index=X.index, columns=names)
        source = {name: name if name in columns else params.MONTH_COL for name in names}
        contrib = raw.T.groupby(source).sum().T
        base = float(model.intercept_)
    elif isinstance(model, xgboost.XGBRegressor):
        values = model.get_booster().predict(xgboost.DMatrix(X), pred_contribs=True)
        contrib = pd.DataFrame(values[:, :-1], index=X.index, columns=columns)
        base = float(values[0, -1])
    else:
        raise ValueError(f"no explanation for model type {type(model).__name__}")

    return contrib.reindex(columns=columns, fill_value=0.0), base


def group_contributions(contrib: pd.DataFrame,
                        groups: dict[str, list[str]] = params.DRIVER_GROUPS) -> pd.DataFrame:
    """Suma las contribuciones por grupo. Los grupos sin features en el modelo no aparecen."""
    out = {}
    for group, features in groups.items():
        present = [f for f in features if f in contrib.columns]
        if present:
            out[group] = contrib[present].sum(axis=1)
    missing = set(contrib.columns) - {f for fs in groups.values() for f in fs}
    if missing:
        out["Otros"] = contrib[sorted(missing)].sum(axis=1)
    return pd.DataFrame(out, index=contrib.index)


def explain_change(pipe: Pipeline, X: pd.DataFrame, month: pd.Timestamp) -> dict:
    """Nowcast de `month`, del mes anterior, y el cambio repartido por grupo.

    `X` debe tener una fila por mes (índice inicio de mes) e incluir el mes anterior.
    """
    month = pd.Timestamp(month)
    previous = month - pd.offsets.MonthBegin(1)
    if month not in X.index or previous not in X.index:
        raise KeyError(f"need rows for {previous:%Y-%m} and {month:%Y-%m}")

    contrib, base = feature_contributions(pipe, X.loc[[previous, month]])
    groups = group_contributions(contrib)
    prediction = base + contrib.sum(axis=1)

    drivers = pd.DataFrame({
        "group": groups.columns,
        "contribution": groups.loc[month].to_numpy(),
        "change": (groups.loc[month] - groups.loc[previous]).to_numpy(),
    }).sort_values("change", key=abs, ascending=False, ignore_index=True)

    return {
        "month": month,
        "previous_month": previous,
        "nowcast": float(prediction.loc[month]),
        "previous_nowcast": float(prediction.loc[previous]),
        "change": float(prediction.loc[month] - prediction.loc[previous]),
        "base": base,
        "drivers": drivers,
    }


def describe_change(explanation: dict, top: int = 2, threshold: float = 0.005) -> str:
    """Frase corta que dice cuánto cambió el nowcast y qué grupos lo movieron."""
    change = explanation["change"]
    drivers = explanation["drivers"]
    month = f"{explanation['month']:%Y-%m}"
    previous = f"{explanation['previous_month']:%Y-%m}"

    if abs(change) < threshold:
        text = f"El nowcast de {month} es prácticamente igual al de {previous}."
    else:
        direction = "sube" if change > 0 else "baja"
        text = f"El nowcast de {month} {direction} {abs(change):.2f} pp frente a {previous}."

    def listing(rows: pd.DataFrame) -> str:
        return ", ".join(f"{row.group} ({row.change:+.2f} pp)" for row in rows.itertuples())

    sign = 1 if change >= 0 else -1
    pushing = drivers[drivers["change"] * sign > threshold].head(top)
    against = drivers[drivers["change"] * sign < -threshold].head(1)
    if not pushing.empty:
        text += f" Lo que más lo movió: {listing(pushing)}."
    if not against.empty:
        text += f" En sentido contrario: {listing(against)}."
    return text
