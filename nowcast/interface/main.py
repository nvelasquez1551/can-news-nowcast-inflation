"""Interfaz: solo orquesta train(), evaluate() y predict(). Sin lógica propia.

    python -m nowcast.interface.main            # train + predict
    python -c "from nowcast.interface.main import train; train(use_news=True)"
                                                # modelos con noticias: linear_news, xgb_news
    RUN_TEST=1 python -m nowcast.interface.main # además evalúa en 2020-2023
"""

import pandas as pd

from nowcast import params
from nowcast.ml_logic.data import load_news_index, load_raw_data, split_by_date
from nowcast.ml_logic.features import build_model_table, get_feature_columns
from nowcast.ml_logic.model import (build_model, compute_metrics, cross_validate,
                                    diebold_mariano, evaluate as evaluate_fixed,
                                    expanding_forecast, grid_search, search_summary)
from nowcast.ml_logic.registry import load_model, save_model


def load_table(use_news: bool | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Tabla del modelo y columnas de features. use_news=None usa params.USE_NEWS."""
    use_news = params.USE_NEWS if use_news is None else use_news
    news = load_news_index() if use_news else None
    table = build_model_table(load_raw_data(), news=news, use_news=use_news)
    return table, get_feature_columns(use_news)


def train(use_news: bool | None = None) -> pd.DataFrame:
    """CV en train (hasta 2019), ajuste final y guardado. No toca el test.

    Con noticias, los modelos se guardan como linear_news y xgb_news, así que
    los modelos macro (linear, xgb) siguen disponibles para comparar.
    """
    use_news = params.USE_NEWS if use_news is None else use_news
    table, cols = load_table(use_news)
    if use_news and not set(params.NEWS_COLS) <= set(table.columns):
        raise FileNotFoundError(f"use_news=True needs {params.NEWS_INDEX_PATH}")
    suffix = params.NEWS_MODEL_SUFFIX if use_news else ""
    train_df, _ = split_by_date(table)
    X, y = train_df[cols], train_df[params.TARGET]

    # Regresión lineal: sin hiperparámetros
    linear_folds = cross_validate(build_model("linear", cols), X, y)
    linear = build_model("linear", cols).fit(X, y)
    save_model(linear, f"linear{suffix}", X, extra={
        "cv_rmse_mean": linear_folds.mean(), "cv_rmse_std": linear_folds.std()})

    # XGBoost: grid search con TimeSeriesSplit
    xgb = search_summary(grid_search(X, y, cols))
    xgb_model = build_model("xgb", cols, xgb["best_params"]).fit(X, y)
    save_model(xgb_model, f"xgb{suffix}", X, xgb["best_params"], extra={
        "cv_rmse_mean": xgb["cv_rmse_mean"], "cv_rmse_std": xgb["cv_rmse_std"]})

    cv_results = pd.DataFrame({
        "model": [f"linear{suffix}", f"xgb{suffix}"],
        "cv_rmse_mean": [linear_folds.mean(), xgb["cv_rmse_mean"]],
        "cv_rmse_std": [linear_folds.std(), xgb["cv_rmse_std"]],
    })
    cv_path = params.CV_RESULTS_NEWS_PATH if use_news else params.CV_RESULTS_PATH
    cv_path.parent.mkdir(parents=True, exist_ok=True)
    cv_results.to_csv(cv_path, index=False)
    print("XGBoost best params:", xgb["best_params"])
    print(cv_results.round(4).to_string(index=False))
    return cv_results


def evaluate() -> dict[str, pd.DataFrame] | None:
    """Evaluación en 2020-2023: fija y expanding refit. Solo con RUN_TEST=1."""
    if not params.RUN_TEST:
        print("RUN_TEST=0: test evaluation skipped (set RUN_TEST=1 to run it)")
        return None

    table, cols = load_table()
    train_df, test_df = split_by_date(table)

    # 1. Evaluación fija: modelos guardados por train() (1999-2019)
    fitted = {}
    for name in params.MODEL_NAMES:
        pipe, metadata = load_model(name)
        assert metadata["features"] == cols, f"{name}: saved features differ"
        fitted[name] = pipe
    fixed_preds, fixed_results = evaluate_fixed(fitted, train_df, test_df, cols)
    fixed_dm = diebold_mariano(fixed_preds)

    # 2. Expanding refit (evaluación principal), retune cada RETUNE_EVERY meses
    expanding = pd.concat(
        [expanding_forecast(name, table, cols, retune_every=params.RETUNE_EVERY)
         .assign(model=name) for name in params.MODEL_NAMES],
        ignore_index=True)
    wide = expanding.pivot(index="month", columns="model", values="prediction")
    first = expanding[expanding["model"] == params.MODEL_NAMES[0]].set_index("month")
    wide = pd.concat(
        [first[["actual"] + params.BASELINES].rename(columns={"actual": "y"}),
         wide], axis=1)
    expanding_results = compute_metrics(wide, params.BASELINES + params.MODEL_NAMES)
    expanding_dm = diebold_mariano(wide)

    outputs = {
        params.FIXED_PREDICTIONS_PATH: fixed_preds.rename_axis("month"),
        params.FIXED_RESULTS_PATH: fixed_results,
        params.FIXED_DM_PATH: fixed_dm,
        params.EXPANDING_PREDICTIONS_PATH: expanding,
        params.EXPANDING_RESULTS_PATH: expanding_results,
        params.EXPANDING_DM_PATH: expanding_dm,
    }
    for path, frame in outputs.items():
        frame.to_csv(path, index=path == params.FIXED_PREDICTIONS_PATH)

    print("Fixed evaluation:\n", fixed_results.round(4).to_string(index=False))
    print("Expanding refit:\n", expanding_results.round(4).to_string(index=False))
    return {"fixed_predictions": fixed_preds, "fixed_results": fixed_results,
            "fixed_dm": fixed_dm, "expanding": expanding,
            "expanding_results": expanding_results, "expanding_dm": expanding_dm}


def predict(X_new: pd.DataFrame | None = None) -> pd.DataFrame:
    """Nowcast con el modelo principal guardado. Devuelve (month, nowcast).

    Sin X_new, predice el último mes de la tabla del modelo.
    """
    pipe, metadata = load_model(params.PRIMARY_MODEL)
    if X_new is None:
        table, _ = load_table()
        X_new = table.iloc[[-1]]
    X_new = X_new[metadata["features"]]
    return pd.DataFrame({"month": X_new.index,
                         "nowcast": pipe.predict(X_new)})


if __name__ == "__main__":
    train()
    evaluate()
    print(predict())
