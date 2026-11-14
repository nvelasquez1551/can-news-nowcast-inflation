"""Tests de nowcast.ml_logic.preprocessor: solo se ajusta con train."""

import numpy as np
import pytest

from nowcast import params
from nowcast.ml_logic.data import split_by_date
from nowcast.ml_logic.model import build_model
from nowcast.ml_logic.preprocessor import build_preprocessor


def test_scaler_learns_train_mean_not_full_sample(table):
    train, test = split_by_date(table)
    pipe = build_model("linear").fit(train[params.FEATURES], train[params.TARGET])
    scaler = pipe.named_steps["preprocessor"].named_transformers_["scale"]

    train_mean = train[params.NUMERIC_FEATURES].mean().to_numpy()
    full_mean = table.loc[:params.TEST_END, params.NUMERIC_FEATURES].mean().to_numpy()
    np.testing.assert_allclose(scaler.mean_, train_mean)
    assert not np.allclose(scaler.mean_, full_mean)


def test_linear_preprocessor_scales_and_one_hot_encodes_month(table):
    train, _ = split_by_date(table)
    pre = build_preprocessor("linear", params.FEATURES).fit(train[params.FEATURES])
    names = list(pre.get_feature_names_out())

    month_cols = [n for n in names if n.startswith("month_")]
    assert month_cols == [f"month_{m}" for m in range(2, 13)]  # enero = base
    assert names[:len(params.NUMERIC_FEATURES)] == params.NUMERIC_FEATURES

    scaled = pre.transform(train[params.FEATURES])[:, :len(params.NUMERIC_FEATURES)]
    np.testing.assert_allclose(scaled.mean(axis=0), 0, atol=1e-10)
    np.testing.assert_allclose(scaled.std(axis=0), 1)


def test_no_month_encoder_without_month_column():
    pre = build_preprocessor("linear", params.NUMERIC_FEATURES)
    assert [name for name, _, _ in pre.transformers] == ["scale"]


def test_xgb_has_no_preprocessor():
    assert build_preprocessor("xgb", params.FEATURES) is None


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        build_preprocessor("rf", params.FEATURES)
