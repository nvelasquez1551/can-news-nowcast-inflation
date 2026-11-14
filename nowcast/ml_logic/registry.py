"""Registry local: guarda y carga modelos (joblib) con su metadata (json).

Cada modelo se guarda como models/<modelo>_<timestamp>.joblib y, al lado,
models/<modelo>_<timestamp>.json con fecha de entrenamiento, periodo de train,
features e hiperparámetros.
"""

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import xgboost
from sklearn.pipeline import Pipeline

from nowcast import params


def save_model(pipe: Pipeline,
               model_name: str,
               X_train: pd.DataFrame,
               model_params: dict | None = None,
               extra: dict | None = None,
               models_dir: Path = params.MODELS_DIR) -> Path:
    """Guarda el Pipeline ajustado y su metadata. Devuelve la ruta del .joblib."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    stem = f"{model_name}_{now.strftime(params.REGISTRY_TIMESTAMP_FORMAT)}"
    model_path = models_dir / f"{stem}.joblib"
    # Nunca sobrescribir un modelo guardado en silencio
    if model_path.exists() or model_path.with_suffix(".json").exists():
        raise FileExistsError(f"{model_path} already exists")

    metadata = {
        "model_name": model_name,
        "trained_at": now.isoformat(timespec="seconds"),
        "train_start": f"{X_train.index.min():%Y-%m}",
        "train_end": f"{X_train.index.max():%Y-%m}",
        "n_train": len(X_train),
        "features": list(X_train.columns),
        "hyperparameters": model_params or {},
        "seed": params.SEED,
        "versions": {"sklearn": sklearn.__version__,
                     "xgboost": xgboost.__version__},
        **(extra or {}),  # p. ej. CV RMSE
    }

    joblib.dump(pipe, model_path)
    model_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, default=str))
    return model_path


def load_model(model_name: str,
               models_dir: Path = params.MODELS_DIR) -> tuple[Pipeline, dict]:
    """Carga el modelo más reciente de `model_name` y su metadata."""
    # El timestamp empieza por un dígito: así "linear" no encuentra "linear_news_..."
    candidates = sorted(Path(models_dir).glob(f"{model_name}_[0-9]*.joblib"))
    if not candidates:
        raise FileNotFoundError(
            f"no saved model '{model_name}' in {models_dir}; run train() first")

    model_path = candidates[-1]  # el timestamp ordena cronológicamente
    pipe = joblib.load(model_path)
    metadata = json.loads(model_path.with_suffix(".json").read_text())
    return pipe, metadata
