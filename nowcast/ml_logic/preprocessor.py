"""Preprocesamiento: SOLO lo que se ajusta con datos de train (notebook 03 v1).

- Regresión lineal: ColumnTransformer = StandardScaler en columnas numéricas
  + OneHotEncoder(drop="first") en `month` (si USE_MONTH).
- XGBoost: sin preprocesamiento (`month` queda numérico, sin escalar).

El preprocesador va DENTRO del Pipeline, así en cada fold de CV y en cada
refit se ajusta solo con las filas de train que recibe.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from nowcast import params


def build_preprocessor(model_name: str,
                       feature_columns: list[str]) -> ColumnTransformer | None:
    """Devuelve el preprocesador sin ajustar para `model_name`, o None si no lleva."""
    if model_name == "xgb":
        return None
    if model_name != "linear":
        raise ValueError(f"unknown model: {model_name}")

    numeric = [c for c in feature_columns if c != params.MONTH_COL]
    transformers = [("scale", StandardScaler(), numeric)]
    if params.MONTH_COL in feature_columns:
        transformers.append(
            ("month", OneHotEncoder(drop="first"), [params.MONTH_COL]))

    return ColumnTransformer(transformers, verbose_feature_names_out=False)
