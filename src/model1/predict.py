"""Single-headline inference for the Streamlit demo."""

from __future__ import annotations

import logging
from typing import Any

from src.model1.config import Model1Config, load_config
from src.model1.score import LoadedScorer, load_both_scorers, score_texts

LOGGER = logging.getLogger(__name__)

_SCORERS: dict[str, LoadedScorer] | None = None


def _scorers(config: Model1Config | None = None) -> dict[str, LoadedScorer]:
    global _SCORERS
    if _SCORERS is None:
        cfg = config or load_config()
        _SCORERS = load_both_scorers(cfg)
    return _SCORERS


def classify_headline(text: str, config: Model1Config | None = None) -> dict[str, Any]:
    """Return both models' raw label, probabilities, and score for one headline."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string.")
    cfg = config or load_config()
    scorers = _scorers(cfg)
    result: dict[str, Any] = {"text": text}
    for key, scorer in scorers.items():
        scored = score_texts(
            [text],
            scorer,
            batch_size=cfg.batch_size,
            max_length=cfg.max_length,
        )
        row = scored.iloc[0]
        if key == "finbert":
            result["finbert"] = {
                "label": str(row["finbert_label"]),
                "role": str(row["finbert_role"]),
                "p_pos": float(row["finbert_p_pos"]),
                "p_neg": float(row["finbert_p_neg"]),
                "p_neu": float(row["finbert_p_neu"]),
                "score": float(row["finbert_score"]),
            }
        else:
            result["inflabert"] = {
                "label": str(row["infla_label"]),
                "role": str(row["infla_role"]),
                "p_up": float(row["infla_p_up"]),
                "p_down": float(row["infla_p_down"]),
                "p_neu": float(row["infla_p_neu"]),
                "score": float(row["infla_score"]),
            }
    return result
