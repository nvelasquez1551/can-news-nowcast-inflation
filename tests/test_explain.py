"""Tests de nowcast.ml_logic.explain: las contribuciones suman la predicción."""

import numpy as np
import pandas as pd
import pytest

from nowcast import params
from nowcast.ml_logic.data import split_by_date
from nowcast.ml_logic.explain import (describe_change, explain_change,
                                      feature_contributions, group_contributions)
from nowcast.ml_logic.model import build_model


@pytest.fixture(params=["linear", "xgb"])
def fitted(request, table):
    train, _ = split_by_date(table)
    hp = {"model__n_estimators": 20, "model__max_depth": 2} if request.param == "xgb" else None
    pipe = build_model(request.param, model_params=hp)
    return pipe.fit(train[params.FEATURES], train[params.TARGET])


def test_base_plus_contributions_equals_prediction(fitted, table):
    X = table[params.FEATURES]
    contrib, base = feature_contributions(fitted, X)
    assert list(contrib.columns) == params.FEATURES  # one-hot de month sumado en `month`
    np.testing.assert_allclose(base + contrib.sum(axis=1), fitted.predict(X), atol=1e-5)


def test_groups_cover_every_feature_once(fitted, table):
    contrib, _ = feature_contributions(fitted, table[params.FEATURES])
    groups = group_contributions(contrib)
    assert "Otros" not in groups.columns
    np.testing.assert_allclose(groups.sum(axis=1), contrib.sum(axis=1), atol=1e-6)


def test_change_is_split_exactly_between_groups(fitted, table):
    X = table[params.FEATURES]
    month = pd.Timestamp("2021-06-01")
    result = explain_change(fitted, X, month)
    preds = pd.Series(fitted.predict(X), index=X.index)

    assert result["nowcast"] == pytest.approx(preds[month], abs=1e-5)
    assert result["previous_nowcast"] == pytest.approx(preds["2021-05-01"], abs=1e-5)
    assert result["drivers"]["change"].sum() == pytest.approx(result["change"], abs=1e-5)
    assert result["drivers"]["change"].abs().is_monotonic_decreasing


def test_explain_change_needs_previous_month(fitted, table):
    with pytest.raises(KeyError):
        explain_change(fitted, table[params.FEATURES], table.index[0])


def test_describe_change_names_main_drivers():
    drivers = pd.DataFrame({"group": ["Gasolina", "Empleo", "Vivienda"],
                            "contribution": [0.3, 0.0, -0.1],
                            "change": [0.20, 0.001, -0.05]})
    text = describe_change({"month": pd.Timestamp("2022-06-01"),
                            "previous_month": pd.Timestamp("2022-05-01"),
                            "change": 0.151, "drivers": drivers})
    assert text.startswith("El nowcast de 2022-06 sube 0.15 pp frente a 2022-05.")
    assert "Gasolina (+0.20 pp)" in text
    assert "En sentido contrario: Vivienda (-0.05 pp)" in text
    assert "Empleo" not in text
