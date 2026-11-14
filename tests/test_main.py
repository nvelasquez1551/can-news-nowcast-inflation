"""Test de punta a punta de nowcast.interface.main con datos sintéticos.

Todo se escribe en una carpeta temporal: no toca data/, models/ ni el test real.
"""

from functools import partial

import pandas as pd
import pytest

from nowcast import params
from nowcast.interface import main
from nowcast.ml_logic import model, registry
from tests.conftest import SMALL_GRID, make_raw

TEST_OUTPUTS = ["FIXED_PREDICTIONS_PATH", "FIXED_RESULTS_PATH", "FIXED_DM_PATH",
                "EXPANDING_PREDICTIONS_PATH", "EXPANDING_RESULTS_PATH",
                "EXPANDING_DM_PATH"]


@pytest.fixture
def sandbox(monkeypatch, tmp_path):
    """Datos sintéticos, modelos y salidas en tmp_path, grilla mínima."""
    monkeypatch.setattr(main, "load_raw_data", make_raw)
    monkeypatch.setattr(params, "USE_NEWS", False)
    monkeypatch.setattr(main, "save_model",
                        partial(registry.save_model, models_dir=tmp_path / "models"))
    monkeypatch.setattr(main, "load_model",
                        partial(registry.load_model, models_dir=tmp_path / "models"))
    small_search = partial(model.grid_search, grid=SMALL_GRID)
    monkeypatch.setattr(main, "grid_search", small_search)
    monkeypatch.setattr(model, "grid_search", small_search)  # usado por expanding
    for name in ["CV_RESULTS_PATH"] + TEST_OUTPUTS:
        monkeypatch.setattr(params, name, tmp_path / f"{name}.csv")
    return tmp_path


def test_flow_without_run_test_never_touches_test(sandbox, monkeypatch):
    monkeypatch.setattr(params, "RUN_TEST", False)

    cv = main.train()
    assert list(cv["model"]) == ["linear", "xgb"]
    assert (cv["cv_rmse_mean"] > 0).all()
    assert len(list((sandbox / "models").glob("*.joblib"))) == 2
    assert params.CV_RESULTS_PATH.exists()

    assert main.evaluate() is None
    assert not any(getattr(params, name).exists() for name in TEST_OUTPUTS)

    nowcast = main.predict()
    assert list(nowcast.columns) == ["month", "nowcast"]
    assert len(nowcast) == 1
    assert nowcast["month"].iloc[0] == main.load_table()[0].index.max()


def test_flow_with_run_test_writes_all_outputs(sandbox, monkeypatch):
    monkeypatch.setattr(params, "RUN_TEST", True)
    main.train()
    out = main.evaluate()

    assert all(getattr(params, name).exists() for name in TEST_OUTPUTS)
    assert len(out["fixed_predictions"]) == params.N_TEST_MONTHS
    assert len(out["expanding"]) == params.N_TEST_MONTHS * len(params.MODEL_NAMES)
    expected_rows = params.BASELINES + params.MODEL_NAMES
    assert list(out["fixed_results"]["model"]) == expected_rows
    assert list(out["expanding_results"]["model"]) == expected_rows
    assert len(out["fixed_dm"]) == len(out["expanding_dm"]) == len(params.DM_PAIRS)
    # Expanding: XGBoost re-busca cada enero, la lineal nunca
    xgb = out["expanding"][out["expanding"]["model"] == "xgb"]
    assert xgb["retuned"].sum() == 4
    assert not out["expanding"].loc[out["expanding"]["model"] == "linear",
                                    "retuned"].any()


def test_predict_with_given_rows(sandbox, monkeypatch):
    monkeypatch.setattr(params, "RUN_TEST", False)
    main.train()
    table, cols = main.load_table()
    rows = table.loc["2010-01-01":"2010-03-01"]
    nowcast = main.predict(rows)
    assert list(nowcast["month"]) == list(rows.index)
    assert nowcast["nowcast"].notna().all()


def test_evaluate_rejects_models_with_other_features(sandbox, monkeypatch):
    monkeypatch.setattr(params, "RUN_TEST", True)
    main.train()
    monkeypatch.setattr(main, "get_feature_columns",
                        lambda use_news: params.NUMERIC_FEATURES)
    with pytest.raises(AssertionError, match="saved features differ"):
        main.evaluate()


def test_train_with_news_saves_separate_models(sandbox, monkeypatch):
    news = make_raw()[[]].assign(**{col: 0.1 for col in params.NEWS_COLS})
    news.iloc[::2] = -0.2
    monkeypatch.setattr(main, "load_news_index", lambda: news)
    monkeypatch.setattr(params, "CV_RESULTS_NEWS_PATH", sandbox / "cv_news.csv")

    cv = main.train(use_news=True)
    assert list(cv["model"]) == ["linear_news", "xgb_news"]
    assert params.CV_RESULTS_NEWS_PATH.exists() and not params.CV_RESULTS_PATH.exists()
    _, meta = main.load_model("linear_news")
    assert meta["features"] == params.FEATURES + params.NEWS_COLS
    with pytest.raises(FileNotFoundError):
        main.load_model("linear")
