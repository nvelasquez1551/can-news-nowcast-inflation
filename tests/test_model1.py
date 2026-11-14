"""Synthetic tests for Model 1. Do not load Hugging Face models or FNSPID."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.model1.aggregate import (
    INDEX_COLUMNS,
    aggregate_monthly,
    apply_start_month,
    find_start_month,
    in_cutoff_window,
)
from src.model1.config import load_config
from src.model1.filter import (
    compile_exclude_pattern,
    compile_keyword_pattern,
    matches_keywords,
    price_related_mask,
)
from src.model1.labels import (
    FINBERT_PROB_COLUMNS,
    assert_valid_probs,
    assert_valid_scores,
    decode_logits,
    resolve_role_map,
    softmax,
)
from src.model1.load import _clean_chunk, _rebuild_seen, write_parquet_atomic

BROAD_KEYWORDS = [
    "inflation",
    "cpi",
    "core cpi",
    "consumer price index",
    "fed",
    "oil",
    "gasoline prices",
]


def test_keyword_filter_true_and_false_examples():
    pattern = compile_keyword_pattern(BROAD_KEYWORDS)
    positives = [
        "Gasoline prices soar",
        "CORE CPI surprises to the upside",
        "The Fed signals a rate pause",
        "Oil inventory draw sparks concern",
        "Consumer price index holds steady",
    ]
    negatives = [
        "Federal funds futures rally",  # "fed" must not match "federal"
        "Foil demand rises at packagers",  # "oil" must not match "foil"
        "Company announces new CEO",
        "",
        None,
    ]
    for text in positives:
        assert matches_keywords(text, pattern), text
    for text in negatives:
        assert not matches_keywords(text, pattern), text

    frame = pd.DataFrame(
        {
            "Article_title": ["Quiet earnings day", "Tech IPO prices above range"],
            "Lsa_summary": ["Inflation fears linger in the outlook.", ""],
        }
    )
    mask = price_related_mask(frame, pattern)
    assert mask.tolist() == [True, False]


def test_probabilities_sum_to_one_and_scores_in_unit_interval():
    logits = np.array(
        [
            [2.0, 0.5, -1.0],
            [0.0, 0.0, 0.0],
            [5.0, -5.0, 0.0],
        ]
    )
    probs = softmax(logits)
    assert_valid_probs(probs)
    assert np.allclose(probs.sum(axis=1), 1.0)
    scores = probs[:, 0] - probs[:, 1]
    assert_valid_scores(scores)
    assert np.all(scores >= -1.0) and np.all(scores <= 1.0)


def test_label_roles_follow_id2label_not_column_position():
    aliases = {
        "pos": ["positive"],
        "neg": ["negative"],
        "neu": ["neutral"],
    }
    # Shuffled vs FinBERT's published 0=positive, 1=negative, 2=neutral.
    id2label = {0: "neutral", 1: "positive", 2: "negative"}
    role_map = resolve_role_map(
        id2label,
        aliases,
        positive_role="pos",
        negative_role="neg",
        model_name="shuffled-finbert",
    )
    assert role_map.role_to_index == {"neu": 0, "pos": 1, "neg": 2}

    # Class 0 logit dominates -> predicted role is neu, not pos.
    logits = np.array([[10.0, 0.0, 0.0]])
    decoded = decode_logits(logits, role_map)
    assert decoded["role"][0] == "neu"
    assert decoded["label"][0] == "neutral"
    assert decoded["by_role"]["neu"][0] > decoded["by_role"]["pos"][0]


def test_inflabert_positive_maps_to_up_even_if_index_order_changes():
    aliases = {
        "up": ["positive", "up"],
        "down": ["negative", "down"],
        "neu": ["neutral"],
    }
    id2label = {0: "positive", 1: "negative", 2: "neutral"}
    role_map = resolve_role_map(
        id2label,
        aliases,
        positive_role="up",
        negative_role="down",
        model_name="shuffled-infla",
    )
    logits = np.array([[0.0, 4.0, 0.0]])  # class 1 = negative = down
    decoded = decode_logits(logits, role_map)
    assert decoded["role"][0] == "down"
    assert decoded["label"][0] == "negative"
    np.testing.assert_allclose(
        decoded["score"][0],
        decoded["by_role"]["up"][0] - decoded["by_role"]["down"][0],
    )


def _article(date: str, title: str, finbert_role: str, infla_role: str, finbert_score: float, infla_score: float):
    return {
        "Date": date,
        "Article_title": title,
        "finbert_role": finbert_role,
        "infla_role": infla_role,
        "finbert_score": finbert_score,
        "infla_score": infla_score,
        "finbert_label": {"pos": "positive", "neg": "negative", "neu": "neutral"}[finbert_role],
        "infla_label": {"up": "positive", "down": "negative", "neu": "neutral"}[infla_role],
    }


def test_cutoff_rule_excludes_day_16_from_current_and_next_month():
    rows = [
        _article("2010-01-15 23:59:00+00:00", "in window", "pos", "up", 0.4, 0.5),
        _article("2010-01-16 00:00:00+00:00", "after cutoff", "neg", "down", -0.9, -0.8),
        _article("2010-02-01 00:00:00+00:00", "next month", "pos", "up", 0.2, 0.1),
    ]
    scored = pd.DataFrame(rows)
    dates = pd.to_datetime(scored["Date"], utc=True)
    mask = in_cutoff_window(dates, cutoff_day=15)
    assert mask.tolist() == [True, False, True]

    monthly = aggregate_monthly(scored, cutoff_day=15)
    jan = monthly.loc[monthly["month"].eq(pd.Timestamp("2010-01-01"))].iloc[0]
    feb = monthly.loc[monthly["month"].eq(pd.Timestamp("2010-02-01"))].iloc[0]
    assert int(jan["news_n_articles"]) == 1
    assert pytest.approx(jan["finbert_score_mean"], rel=1e-6) == 0.4
    assert int(feb["news_n_articles"]) == 1
    assert pytest.approx(feb["finbert_score_mean"], rel=1e-6) == 0.2
    # Day-16 January article is in neither month.
    assert jan["news_n_articles"] + feb["news_n_articles"] == 2


def test_start_date_rule_skips_early_broken_streak():
    months = pd.date_range("2010-01-01", "2011-06-01", freq="MS")
    counts = []
    for month in months:
        n = 30
        if month == pd.Timestamp("2010-06-01"):
            n = 10  # breaks the first streak
        counts.append({"month": month, "news_n_articles": n})
    monthly = pd.DataFrame(counts)
    start = find_start_month(monthly, min_articles_per_month=30, consecutive_months=12)
    assert start == pd.Timestamp("2010-07-01")


def test_aggregation_matches_hand_computed_values():
    rows = [
        _article("2010-03-01 12:00:00+00:00", "a", "pos", "up", 0.5, 1.0),
        _article("2010-03-10 12:00:00+00:00", "b", "neg", "down", -0.5, -1.0),
        _article("2010-03-15 12:00:00+00:00", "c", "neu", "neu", 0.0, 0.0),
        _article("2010-03-20 12:00:00+00:00", "too late", "pos", "up", 1.0, 1.0),
    ]
    monthly = aggregate_monthly(pd.DataFrame(rows), cutoff_day=15)
    mar = monthly.loc[monthly["month"].eq(pd.Timestamp("2010-03-01"))].iloc[0]
    assert int(mar["news_n_articles"]) == 3
    assert pytest.approx(mar["finbert_score_mean"], rel=1e-9) == 0.0
    assert pytest.approx(mar["infla_score_mean"], rel=1e-9) == 0.0
    assert pytest.approx(mar["finbert_share_pos"], rel=1e-9) == 1 / 3
    assert pytest.approx(mar["finbert_share_neg"], rel=1e-9) == 1 / 3
    assert pytest.approx(mar["finbert_net_share"], rel=1e-9) == 0.0
    assert pytest.approx(mar["infla_share_up"], rel=1e-9) == 1 / 3
    assert pytest.approx(mar["infla_share_down"], rel=1e-9) == 1 / 3
    assert pytest.approx(mar["infla_net_share"], rel=1e-9) == 0.0
    assert list(monthly.columns) == INDEX_COLUMNS


def test_empty_month_stays_nan_not_forward_filled():
    rows = [
        _article("2010-01-02 00:00:00+00:00", "jan", "pos", "up", 0.3, 0.4),
        _article("2010-02-20 00:00:00+00:00", "feb after cutoff only", "neg", "down", -0.2, -0.1),
        _article("2010-03-05 00:00:00+00:00", "mar", "neg", "down", -0.1, -0.2),
    ]
    monthly = aggregate_monthly(pd.DataFrame(rows), cutoff_day=15)
    months = list(pd.to_datetime(monthly["month"]))
    assert months == [
        pd.Timestamp("2010-01-01"),
        pd.Timestamp("2010-02-01"),
        pd.Timestamp("2010-03-01"),
    ]
    feb = monthly.loc[monthly["month"].eq(pd.Timestamp("2010-02-01"))].iloc[0]
    jan = monthly.loc[monthly["month"].eq(pd.Timestamp("2010-01-01"))].iloc[0]
    assert int(feb["news_n_articles"]) == 0
    assert pd.isna(feb["finbert_score_mean"])
    assert pd.isna(feb["infla_score_mean"])
    assert pytest.approx(jan["finbert_score_mean"], rel=1e-9) == 0.3
    assert pd.isna(feb["finbert_score_mean"])  # still NaN after January exists


def test_apply_start_month_keeps_later_gaps():
    months = pd.date_range("2010-01-01", "2011-03-01", freq="MS")
    monthly = pd.DataFrame(
        {
            "month": months,
            "finbert_score_mean": 0.1,
            "finbert_share_pos": 0.2,
            "finbert_share_neg": 0.1,
            "finbert_net_share": 0.1,
            "infla_score_mean": 0.0,
            "infla_share_up": 0.2,
            "infla_share_down": 0.2,
            "infla_net_share": 0.0,
            "news_n_articles": [30] * len(months),
        }
    )
    monthly.loc[monthly["month"].eq(pd.Timestamp("2011-02-01")), "news_n_articles"] = 5
    start = pd.Timestamp("2010-01-01")
    index, full = apply_start_month(monthly, start, min_articles_per_month=30)
    assert index["month"].min() == start
    flagged = full.loc[full["below_min_flag"], "month"].tolist()
    assert flagged == [pd.Timestamp("2011-02-01")]
    assert 5 in set(index["news_n_articles"].tolist())


EXCLUDE_PATTERNS = load_config().exclude_title_patterns


def test_exclude_patterns_drop_offering_headlines_only():
    pattern = compile_keyword_pattern(["prices", "inflation"])
    exclude = compile_exclude_pattern(EXCLUDE_PATTERNS)
    frame = pd.DataFrame(
        {
            "Article_title": [
                "Apple Prices $5 Billion Senior Notes Offering",
                "XYZ prices public offering of common stock",
                "Tech IPO prices above range",
                "Gasoline prices soar",
                "Consumer prices rise; Treasury notes rally",
            ],
            "Lsa_summary": ["Inflation risk noted.", "", "", "", ""],
        }
    )
    mask = price_related_mask(frame, pattern, exclude=exclude)
    assert mask.tolist() == [False, False, False, True, True]
    assert compile_exclude_pattern([]) is None


def test_cutoff_31_keeps_the_whole_month():
    rows = [
        _article("2010-01-31 23:59:00+00:00", "last day of january", "pos", "up", 0.4, 0.5),
        _article("2010-02-28 12:00:00+00:00", "last day of february", "neg", "down", -0.2, -0.3),
    ]
    monthly = aggregate_monthly(pd.DataFrame(rows), cutoff_day=31)
    assert monthly["news_n_articles"].tolist() == [1, 1]


def _raw_chunk(rows):
    return pd.DataFrame(
        rows, columns=["Date", "Article_title", "Stock_symbol", "Publisher", "Lsa_summary"]
    )


def test_clean_chunk_dedupes_only_price_related_rows():
    pattern = compile_keyword_pattern(["inflation"])
    seen: set[tuple[str, str]] = set()
    chunk = _raw_chunk(
        [
            ["2010-01-05 10:00:00 UTC", "Inflation picks up", "A", "p", ""],
            ["2010-01-05 18:00:00 UTC", "Inflation picks up", "B", "p", ""],
            ["2010-01-05 10:00:00 UTC", "Earnings beat", "A", "p", ""],
            ["2010-01-05 10:00:00 UTC", "Earnings beat", "B", "p", ""],
            [None, "Inflation without a date", "A", "p", ""],
        ]
    )
    cleaned, stats, _, _ = _clean_chunk(
        chunk, relative_file="f.csv", pattern=pattern, exclude=None, seen=seen
    )
    assert cleaned["Article_title"].tolist() == ["Inflation picks up"]
    assert stats["n_missing_date_or_title"] == 1
    assert stats["n_before_filter"] == 4
    assert stats["n_duplicates"] == 1
    assert stats["n_after_filter"] == 1
    assert seen == {("2010-01-05", "Inflation picks up")}


def test_rebuild_seen_ignores_chunk_not_yet_in_manifest(tmp_path):
    cfg = load_config(data_root=tmp_path, reports_dir=tmp_path / "reports")
    out_dir = cfg.filtered_dir / "nasdaq"
    out_dir.mkdir(parents=True)
    dates = pd.to_datetime(["2010-01-05", "2010-01-06"], utc=True)
    for idx, (date, title) in enumerate(zip(dates, ["committed", "written before crash"])):
        write_parquet_atomic(
            pd.DataFrame({"Date": [date], "Article_title": [title]}),
            out_dir / f"chunk_{idx:05d}.parquet",
        )
    (out_dir / "manifest.json").write_text(
        '{"sample_rows": null, "chunks_written": 1}', encoding="utf-8"
    )
    assert _rebuild_seen(cfg) == {("2010-01-05", "committed")}
    assert not list(out_dir.glob("*.tmp"))


def test_score_texts_restores_input_order_after_length_sort():
    torch = pytest.importorskip("torch")
    from src.model1.score import LoadedScorer, score_texts

    class WordTokenizer:
        def __call__(self, texts, **_):
            width = max(len(text.split()) for text in texts)
            ids = [[1] * len(text.split()) + [0] * (width - len(text.split())) for text in texts]
            return {"input_ids": torch.tensor(ids, dtype=torch.float32)}

    class CountModel:
        def __call__(self, input_ids):
            n_words = input_ids.sum(dim=1, keepdim=True)
            logits = torch.cat([n_words, torch.zeros_like(n_words), torch.zeros_like(n_words)], dim=1)
            return type("Out", (), {"logits": logits})()

    role_map = resolve_role_map(
        {0: "positive", 1: "negative", 2: "neutral"},
        {"pos": ["positive"], "neg": ["negative"], "neu": ["neutral"]},
        positive_role="pos",
        negative_role="neg",
    )
    scorer = LoadedScorer(
        spec=None,
        tokenizer=WordTokenizer(),
        model=CountModel(),
        role_map=role_map,
        device=torch.device("cpu"),
        prefix="finbert",
        prob_columns=FINBERT_PROB_COLUMNS,
    )
    texts = ["one two three four", "one", "one two three", "one two"]
    scored = score_texts(texts, scorer, batch_size=2, max_length=16)
    word_counts = [len(text.split()) for text in texts]
    # More words -> larger positive logit -> larger score, in the original order.
    assert np.argsort(scored["finbert_score"].to_numpy()).tolist() == np.argsort(word_counts).tolist()
    assert len(scored) == len(texts)
