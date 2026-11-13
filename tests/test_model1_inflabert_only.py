"""InflaBERT-only path (models.finbert.enabled=false). No Hugging Face downloads."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.model1 import score as score_mod
from src.model1.aggregate import (
    INDEX_COLUMNS,
    aggregate_monthly,
    apply_start_month,
    index_columns_for,
)
from src.model1.config import DEFAULT_CONFIG_PATH, load_config

INFLA_ONLY_COLUMNS = [
    "month",
    "infla_score_mean",
    "infla_share_up",
    "infla_share_down",
    "infla_net_share",
    "news_n_articles",
]


def _config(tmp_path: Path, *, finbert_enabled: bool | None):
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    if finbert_enabled is None:
        raw["models"]["finbert"].pop("enabled", None)
    else:
        raw["models"]["finbert"]["enabled"] = finbert_enabled
    path = tmp_path / "model1.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_config(path, data_root=tmp_path / "data")


def _infla_article(date: str, role: str, score: float) -> dict:
    return {
        "Date": date,
        "Article_title": "headline",
        "infla_role": role,
        "infla_score": score,
        "infla_label": {"up": "positive", "down": "negative", "neu": "neutral"}[role],
    }


def test_finbert_enabled_flag_defaults_true_and_parses_false(tmp_path):
    assert load_config().finbert_enabled is True  # repo config: unchanged behavior
    assert _config(tmp_path, finbert_enabled=None).finbert_enabled is True
    assert _config(tmp_path, finbert_enabled=False).finbert_enabled is False


def test_load_both_scorers_skips_finbert_when_disabled(tmp_path, monkeypatch):
    loaded = []

    def fake_load_scorer(spec, *, device, prefix, prob_columns):
        loaded.append(prefix)
        return prefix

    monkeypatch.setattr(score_mod, "load_scorer", fake_load_scorer)
    off = score_mod.load_both_scorers(_config(tmp_path, finbert_enabled=False), device="cpu")
    assert list(off) == ["inflabert"] and loaded == ["infla"]

    loaded.clear()
    on = score_mod.load_both_scorers(_config(tmp_path, finbert_enabled=True), device="cpu")
    assert list(on) == ["finbert", "inflabert"] and loaded == ["finbert", "infla"]


def test_score_frame_ignores_finbert_scorer_when_disabled(tmp_path, monkeypatch):
    def fake_score_texts(texts, scorer, *, batch_size, max_length):
        return pd.DataFrame({f"{scorer}_score": [0.5] * len(texts)})

    monkeypatch.setattr(score_mod, "score_texts", fake_score_texts)
    frame = pd.DataFrame({"Date": ["2020-01-02"], "Article_title": ["Inflation rises"]})
    scorers = {"finbert": "finbert", "inflabert": "infla"}

    off = score_mod.score_frame(frame, scorers, _config(tmp_path, finbert_enabled=False))
    assert "infla_score" in off.columns and "finbert_score" not in off.columns

    on = score_mod.score_frame(frame, scorers, _config(tmp_path, finbert_enabled=True))
    assert {"infla_score", "finbert_score"} <= set(on.columns)

    cols = score_mod.active_score_columns(_config(tmp_path, finbert_enabled=False))
    assert cols and all(c.startswith("infla_") for c in cols)


def test_aggregate_inflabert_only_hand_computed():
    rows = [
        _infla_article("2010-03-01 12:00:00+00:00", "up", 1.0),
        _infla_article("2010-03-10 12:00:00+00:00", "up", 0.5),
        _infla_article("2010-03-15 12:00:00+00:00", "down", -0.3),
        _infla_article("2010-03-20 12:00:00+00:00", "down", -1.0),  # after cutoff
    ]
    scored = pd.DataFrame(rows)
    assert index_columns_for(scored) == INFLA_ONLY_COLUMNS

    monthly = aggregate_monthly(scored, cutoff_day=15)
    assert list(monthly.columns) == INFLA_ONLY_COLUMNS
    mar = monthly.iloc[0]
    assert int(mar["news_n_articles"]) == 3
    assert mar["infla_score_mean"] == pytest.approx(0.4)
    assert mar["infla_share_up"] == pytest.approx(2 / 3)
    assert mar["infla_share_down"] == pytest.approx(1 / 3)
    assert mar["infla_net_share"] == pytest.approx(1 / 3)


def test_apply_start_month_inflabert_only_keeps_infla_columns():
    months = pd.date_range("2010-01-01", "2010-12-01", freq="MS")
    monthly = pd.DataFrame(
        {
            "month": months,
            "infla_score_mean": 0.1,
            "infla_share_up": 0.3,
            "infla_share_down": 0.2,
            "infla_net_share": 0.1,
            "news_n_articles": 40,
        }
    )
    index, full = apply_start_month(monthly, months[0], min_articles_per_month=30)
    assert list(index.columns) == INFLA_ONLY_COLUMNS
    assert len(index) == 12 and not full["below_min_flag"].any()


def test_both_families_still_give_full_index_columns():
    scored = pd.DataFrame(
        {
            "Date": ["2010-01-02 00:00:00+00:00"],
            "finbert_score": [0.1],
            "finbert_role": ["pos"],
            "infla_score": [0.2],
            "infla_role": ["up"],
        }
    )
    assert index_columns_for(scored) == INDEX_COLUMNS
