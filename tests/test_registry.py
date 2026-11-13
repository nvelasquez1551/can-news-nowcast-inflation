"""Tests de nowcast.ml_logic.registry: guardar y cargar modelos con metadata."""

import numpy as np
import pytest

from nowcast import params
from nowcast.ml_logic.data import split_by_date
from nowcast.ml_logic.model import build_model
from nowcast.ml_logic.registry import load_model, save_model


def test_save_and_load_roundtrip(table, tmp_path):
    train, _ = split_by_date(table)
    X = train[params.FEATURES]
    hp = {"model__max_depth": 1, "model__n_estimators": 5}
    pipe = build_model("xgb", model_params=hp).fit(X, train["y"])

    save_model(pipe, "xgb", X, hp, extra={"cv_rmse_mean": 0.5}, models_dir=tmp_path)
    loaded, meta = load_model("xgb", models_dir=tmp_path)

    np.testing.assert_array_equal(loaded.predict(X), pipe.predict(X))
    assert meta["features"] == params.FEATURES
    assert meta["hyperparameters"] == hp
    assert meta["train_end"] == "2019-12" and meta["n_train"] == len(X)
    assert meta["cv_rmse_mean"] == 0.5


def test_load_returns_latest_and_saves_never_collide(table, tmp_path):
    train, _ = split_by_date(table)
    X = train[params.FEATURES]
    pipe = build_model("linear").fit(X, train["y"])
    paths = [save_model(pipe, "linear", X, models_dir=tmp_path) for _ in range(5)]

    assert len(set(paths)) == 5
    _, meta = load_model("linear", models_dir=tmp_path)
    assert sorted(paths)[-1] == paths[-1]
    assert meta["model_name"] == "linear"


def test_load_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model("linear", models_dir=tmp_path)
