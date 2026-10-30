"""Keyword filter for price-related headlines and summaries."""

from __future__ import annotations

import re

import pandas as pd


def compile_keyword_pattern(keywords: list[str]) -> re.Pattern[str]:
    """Whole-word, case-insensitive alternation. Longer phrases first."""
    cleaned = [kw.strip() for kw in keywords if str(kw).strip()]
    if not cleaned:
        raise ValueError("Keyword list is empty.")
    cleaned.sort(key=len, reverse=True)
    parts = [rf"\b{re.escape(kw)}\b" for kw in cleaned]
    return re.compile("|".join(parts), flags=re.IGNORECASE)


def matches_keywords(text: object, pattern: re.Pattern[str]) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    return pattern.search(text) is not None


def price_related_mask(
    frame: pd.DataFrame,
    pattern: re.Pattern[str],
    *,
    title_col: str = "Article_title",
    summary_col: str = "Lsa_summary",
) -> pd.Series:
    """True when the title or the LSA summary matches the lexicon."""
    titles = frame[title_col] if title_col in frame.columns else pd.Series("", index=frame.index)
    if summary_col in frame.columns:
        summaries = frame[summary_col]
    else:
        summaries = pd.Series("", index=frame.index)
    title_hit = titles.map(lambda text: matches_keywords(text, pattern))
    summary_hit = summaries.map(lambda text: matches_keywords(text, pattern))
    return title_hit | summary_hit
