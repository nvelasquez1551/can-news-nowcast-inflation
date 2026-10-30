"""Step 4–5: day-cutoff monthly aggregation and the data-driven start month."""

from __future__ import annotations

import logging

import pandas as pd

from src.model1.config import Model1Config

LOGGER = logging.getLogger(__name__)

INDEX_COLUMNS = [
    "month",
    "finbert_score_mean",
    "finbert_share_pos",
    "finbert_share_neg",
    "finbert_net_share",
    "infla_score_mean",
    "infla_share_up",
    "infla_share_down",
    "infla_net_share",
    "news_n_articles",
]


def in_cutoff_window(dates: pd.Series, cutoff_day: int) -> pd.Series:
    """Month t uses UTC days 1 through cutoff_day inclusive. Later days are dropped, not rolled forward."""
    parsed = pd.to_datetime(dates, utc=True, errors="coerce")
    return parsed.dt.day.le(int(cutoff_day))


def month_start(dates: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(dates, utc=True, errors="coerce")
    naive = parsed.dt.tz_convert(None)
    return naive.dt.to_period("M").dt.to_timestamp()


def _month_starts_between(first: pd.Timestamp, last: pd.Timestamp) -> pd.DatetimeIndex:
    start = pd.Timestamp(first).to_period("M").to_timestamp()
    end = pd.Timestamp(last).to_period("M").to_timestamp()
    return pd.date_range(start, end, freq="MS")


def aggregate_monthly(scored: pd.DataFrame, cutoff_day: int) -> pd.DataFrame:
    """One row per calendar month. Empty in-window months stay NaN (no forward fill)."""
    if scored.empty:
        empty = pd.DataFrame(columns=INDEX_COLUMNS + ["below_min_flag"])
        return empty

    work = scored.copy()
    work["Date"] = pd.to_datetime(work["Date"], utc=True, errors="coerce")
    work = work.loc[work["Date"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=INDEX_COLUMNS + ["below_min_flag"])

    work["in_window"] = in_cutoff_window(work["Date"], cutoff_day)
    work["month"] = pd.to_datetime(month_start(work["Date"]))

    window = work.loc[work["in_window"]].copy()
    calendar = _month_starts_between(work["month"].min(), work["month"].max())

    if window.empty:
        monthly = pd.DataFrame({"month": calendar, "news_n_articles": 0})
        for column in INDEX_COLUMNS:
            if column not in monthly.columns:
                monthly[column] = pd.NA
        monthly["news_n_articles"] = 0
        n_dropped = int((~work["in_window"]).sum())
        LOGGER.info(
            "Cutoff day %s dropped %s articles after the window; no in-window rows.",
            cutoff_day,
            n_dropped,
        )
        return monthly.loc[:, INDEX_COLUMNS]

    grouped = window.groupby("month", sort=True)
    agg = grouped.agg(
        finbert_score_mean=("finbert_score", "mean"),
        infla_score_mean=("infla_score", "mean"),
        news_n_articles=("Date", "size"),
    ).reset_index()
    shares = (
        window.assign(
            finbert_share_pos=window["finbert_role"].eq("pos"),
            finbert_share_neg=window["finbert_role"].eq("neg"),
            infla_share_up=window["infla_role"].eq("up"),
            infla_share_down=window["infla_role"].eq("down"),
        )
        .groupby("month", sort=True)[
            [
                "finbert_share_pos",
                "finbert_share_neg",
                "infla_share_up",
                "infla_share_down",
            ]
        ]
        .mean()
        .reset_index()
    )
    agg = agg.merge(shares, on="month", how="left")
    agg["finbert_net_share"] = agg["finbert_share_pos"] - agg["finbert_share_neg"]
    agg["infla_net_share"] = agg["infla_share_up"] - agg["infla_share_down"]

    monthly = pd.DataFrame({"month": calendar}).merge(agg, on="month", how="left")
    monthly["news_n_articles"] = monthly["news_n_articles"].fillna(0).astype(int)
    empty_months = monthly.loc[monthly["news_n_articles"].eq(0), "month"]
    if len(empty_months):
        LOGGER.info(
            "Months with zero in-window articles (left as NaN, not forward-filled): %s",
            [pd.Timestamp(ts).strftime("%Y-%m-%d") for ts in empty_months],
        )

    n_dropped = int((~work["in_window"]).sum())
    LOGGER.info("Cutoff day %s dropped %s articles after the window.", cutoff_day, n_dropped)
    return monthly.loc[:, INDEX_COLUMNS]


def find_start_month(
    monthly: pd.DataFrame,
    *,
    min_articles_per_month: int,
    consecutive_months: int,
) -> pd.Timestamp | None:
    """First month S such that S and the next consecutive_months-1 months each have enough articles."""
    if monthly.empty:
        return None
    ordered = monthly.sort_values("month").reset_index(drop=True)
    counts = ordered["news_n_articles"].fillna(0).astype(int).to_numpy()
    months = pd.to_datetime(ordered["month"])
    window = int(consecutive_months)
    if window <= 0:
        raise ValueError("consecutive_months must be positive.")
    for i in range(0, len(counts) - window + 1):
        if (counts[i : i + window] >= int(min_articles_per_month)).all():
            return pd.Timestamp(months.iloc[i])
    return None


def apply_start_month(
    monthly: pd.DataFrame,
    start: pd.Timestamp,
    *,
    min_articles_per_month: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop months before S from the Model 2 interface; keep the full table with flags."""
    work = monthly.copy()
    work["month"] = pd.to_datetime(work["month"])
    work["below_min_flag"] = False
    after = work["month"] >= pd.Timestamp(start)
    work.loc[after, "below_min_flag"] = work.loc[after, "news_n_articles"].lt(
        int(min_articles_per_month)
    )
    flagged = work.loc[work["below_min_flag"], "month"]
    if len(flagged):
        LOGGER.info(
            "Months after S=%s with fewer than %s articles (kept, not used to move S): %s",
            pd.Timestamp(start).strftime("%Y-%m-%d"),
            min_articles_per_month,
            [pd.Timestamp(ts).strftime("%Y-%m-%d") for ts in flagged],
        )
    index = work.loc[after, INDEX_COLUMNS].copy()
    return index.reset_index(drop=True), work.reset_index(drop=True)


def build_news_index(scored: pd.DataFrame, cfg: Model1Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    monthly = aggregate_monthly(scored, cfg.cutoff_day)
    start = find_start_month(
        monthly,
        min_articles_per_month=cfg.min_articles_per_month,
        consecutive_months=cfg.consecutive_months,
    )
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    if start is None:
        monthly.to_csv(cfg.index_full_path, index=False)
        raise RuntimeError(
            "No start month S: there is no stretch of "
            f"{cfg.consecutive_months} consecutive months with at least "
            f"{cfg.min_articles_per_month} in-window articles. "
            f"Wrote diagnostic counts to {cfg.index_full_path}."
        )
    index, full = apply_start_month(
        monthly, start, min_articles_per_month=cfg.min_articles_per_month
    )
    index.to_csv(cfg.index_csv_path, index=False)
    index.to_parquet(cfg.index_parquet_path, index=False)
    full.to_csv(cfg.index_full_path, index=False)
    cfg.start_month_path.write_text(pd.Timestamp(start).strftime("%Y-%m-%d"), encoding="utf-8")
    LOGGER.info(
        "Start month S=%s (InflaBERT paper starts 2010-01). Index rows=%s.",
        pd.Timestamp(start).strftime("%Y-%m-%d"),
        len(index),
    )
    return index, full, pd.Timestamp(start)
