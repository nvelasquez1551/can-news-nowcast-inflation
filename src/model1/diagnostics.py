"""Write Model 1 diagnostics, data notes, sanity-check sample, and index plot."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.model1.config import Model1Config

LOGGER = logging.getLogger(__name__)

INFLABERT_PAPER_START = "2010-01-01"


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def fetch_cpi_mom(cfg: Model1Config) -> tuple[pd.DataFrame | None, str | None]:
    """Monthly CPI inflation = percent change in FRED CPIAUCSL. Failures are non-fatal."""
    try:
        cpi = pd.read_csv(cfg.cpi_url)
    except Exception as exc:  # network / schema issues must not kill the pipeline
        LOGGER.warning("Could not fetch CPI from %s: %s", cfg.cpi_url, exc)
        return None, str(exc)
    if cfg.cpi_date_col not in cpi.columns or cfg.cpi_value_col not in cpi.columns:
        msg = f"CPI CSV missing {cfg.cpi_date_col}/{cfg.cpi_value_col}; columns={list(cpi.columns)}"
        LOGGER.warning(msg)
        return None, msg
    cpi = cpi.rename(columns={cfg.cpi_date_col: "month", cfg.cpi_value_col: "cpi"})
    cpi["month"] = pd.to_datetime(cpi["month"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    cpi["cpi"] = pd.to_numeric(cpi["cpi"], errors="coerce")
    cpi = cpi.dropna(subset=["month", "cpi"]).sort_values("month")
    cpi["cpi_mom"] = cpi["cpi"].pct_change() * 100.0
    return cpi.loc[:, ["month", "cpi", "cpi_mom"]], None


def _label_distribution(scored: pd.DataFrame, label_col: str) -> pd.DataFrame:
    work = scored.copy()
    work["Date"] = pd.to_datetime(work["Date"], utc=True, errors="coerce")
    work["year"] = work["Date"].dt.year
    counts = (
        work.dropna(subset=["year"])
        .groupby(["year", label_col], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    return counts


def plot_indexes(
    index: pd.DataFrame,
    cpi: pd.DataFrame | None,
    dest: Path,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    plot_df = index.copy()
    plot_df["month"] = pd.to_datetime(plot_df["month"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(plot_df["month"], plot_df["finbert_score_mean"], label="finbert_score_mean")
    ax.plot(plot_df["month"], plot_df["infla_score_mean"], label="infla_score_mean")
    ax.set_ylabel("News index (mean score)")
    ax.set_xlabel("Month")
    ax.axhline(0.0, linewidth=0.8, color="gray")
    if cpi is not None and not cpi.empty:
        ax2 = ax.twinx()
        merged = plot_df.merge(cpi, on="month", how="left")
        ax2.plot(
            merged["month"],
            merged["cpi_mom"],
            linestyle="--",
            color="black",
            alpha=0.6,
            label="CPI mom %",
        )
        ax2.set_ylabel("CPI month-over-month %")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="upper left")
    else:
        ax.legend(loc="upper left")
    ax.set_title("Monthly news indexes vs CPI inflation")
    fig.tight_layout()
    fig.savefig(dest, dpi=150)
    plt.close(fig)


def write_sanity_sample(scored: pd.DataFrame, cfg: Model1Config) -> Path:
    dest = cfg.reports_dir / "sanity_sample.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "Date",
        "Article_title",
        "finbert_label",
        "finbert_role",
        "finbert_score",
        "infla_label",
        "infla_role",
        "infla_score",
    ]
    available = [col for col in cols if col in scored.columns]
    n = min(int(cfg.sanity_sample_size), len(scored))
    if n == 0:
        sample = pd.DataFrame(columns=available + ["human_price_direction"])
    else:
        sample = scored.sample(n=n, random_state=int(cfg.sanity_sample_seed)).loc[:, available]
        sample = sample.sort_values("Date") if "Date" in sample.columns else sample
    sample["human_price_direction"] = ""
    sample.to_csv(dest, index=False)
    LOGGER.info("Wrote %s-row sanity sample to %s (human column left blank).", n, dest)
    return dest


def _md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._\n"
    columns = [str(col) for col in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in frame.itertuples(index=False):
        cells = []
        for value in row:
            if pd.isna(value):
                cells.append("")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_data_notes(
    cfg: Model1Config,
    *,
    start: pd.Timestamp,
    full: pd.DataFrame,
    load_stats: dict,
) -> Path:
    cfg.data_notes_path.parent.mkdir(parents=True, exist_ok=True)
    files = load_stats.get("files", [])
    lines = [
        "# FNSPID data notes (Model 1)",
        "",
        "Filled by the pipeline on the first Colab run. Re-run overwrites this file.",
        "",
        f"- Dataset: `{cfg.dataset_name}` revision `{cfg.dataset_revision}`",
        f"- sample_rows: `{cfg.sample_rows}`",
        f"- Start month S: **{pd.Timestamp(start).strftime('%Y-%m-%d')}** "
        f"(InflaBERT paper starts {INFLABERT_PAPER_START})",
        "",
        "## Files",
        "",
    ]
    for item in files:
        header = item.get("header_columns", [])
        lines.extend(
            [
                f"### `{item.get('source_file')}`",
                "",
                f"- Header columns: `{header}`",
                f"- Raw rows read: {item.get('n_raw')}",
                f"- Dropped missing date or title: {item.get('n_missing_date_or_title')}",
                f"- Dropped exact duplicate (date, title): {item.get('n_duplicates')}",
                f"- After missing/dedup (before keyword filter): {item.get('n_before_filter')}",
                f"- After keyword filter: {item.get('n_after_filter')}",
                "",
                "Year coverage (this file):",
                "",
            ]
        )
        year_rows = []
        for year, counts in sorted(
            (item.get("year_counts") or {}).items(), key=lambda kv: int(kv[0])
        ):
            year_rows.append(
                {
                    "year": year,
                    "n_before_filter": counts.get("before", 0),
                    "n_after_filter": counts.get("after", 0),
                }
            )
        lines.append(_md_table(pd.DataFrame(year_rows)))
        lines.append("")

    flagged = (
        full.loc[full["below_min_flag"]]
        if "below_min_flag" in full.columns
        else full.iloc[0:0]
    )
    lines.extend(
        [
            "## Start-date rule",
            "",
            f"S is the first month whose own in-window count and the next "
            f"{cfg.consecutive_months - 1} months are all "
            f">= {cfg.min_articles_per_month}.",
            "",
            f"Months after S below the threshold (kept in the index): "
            f"{len(flagged)}.",
            "",
        ]
    )
    cfg.data_notes_path.write_text("\n".join(lines), encoding="utf-8")
    return cfg.data_notes_path


def write_diagnostics_report(
    cfg: Model1Config,
    *,
    scored: pd.DataFrame,
    index: pd.DataFrame,
    full: pd.DataFrame,
    start: pd.Timestamp,
    cpi: pd.DataFrame | None,
    cpi_error: str | None,
    plot_path: Path,
    sanity_path: Path,
    load_stats: dict,
) -> Path:
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    dest = cfg.reports_dir / "model1_diagnostics.md"
    year_counts_path = cfg.year_counts_path
    if year_counts_path.exists():
        year_counts = pd.read_csv(year_counts_path)
    else:
        year_counts = pd.DataFrame(columns=["year", "n_before_filter", "n_after_filter"])

    if index.empty or "finbert_score_mean" not in index.columns:
        corr = float("nan")
    else:
        corr = index[["finbert_score_mean", "infla_score_mean"]].corr().iloc[0, 1]
    flagged = full.loc[full["below_min_flag"] == True] if "below_min_flag" in full.columns else full.iloc[0:0]  # noqa: E712
    finbert_labels = _label_distribution(scored, "finbert_label") if not scored.empty else pd.DataFrame()
    infla_labels = _label_distribution(scored, "infla_label") if not scored.empty else pd.DataFrame()
    paper_start = pd.Timestamp(INFLABERT_PAPER_START)
    delta_months = (pd.Timestamp(start).to_period("M") - paper_start.to_period("M")).n

    cpi_note = (
        f"CPI overlay loaded from `{cfg.cpi_url}`."
        if cpi is not None
        else f"CPI overlay missing ({cpi_error}). News indexes were still plotted."
    )

    lines = [
        "# Model 1 diagnostics",
        "",
        f"- Start month **S** = `{pd.Timestamp(start).strftime('%Y-%m-%d')}`",
        f"- InflaBERT paper start = `{INFLABERT_PAPER_START}` ({delta_months:+d} months vs S)",
        f"- Cutoff day = `{cfg.cutoff_day}` (UTC, inclusive)",
        f"- Keywords = `{'narrow' if cfg.use_narrow_lexicon else 'broad'}` "
        f"({len(cfg.keywords)} terms)",
        f"- Scoring text = `{'Lsa_summary when non-empty' if cfg.prefer_lsa_summary else 'headline'}`",
        f"- sample_rows = `{cfg.sample_rows}`",
        f"- Filtered articles scored = `{len(scored)}`",
        f"- Index months (from S) = `{len(index)}`",
        f"- Pearson corr(finbert_score_mean, infla_score_mean) = `{corr:.4f}`"
        if pd.notna(corr)
        else "- Pearson corr(finbert_score_mean, infla_score_mean) = `NA`",
        f"- {cpi_note}",
        f"- Plot: `{plot_path}`",
        f"- Sanity sample: `{sanity_path}` (human labels blank; fill by hand)",
        "",
        "## Articles per year before / after keyword filter",
        "",
        _md_table(year_counts),
        "",
        "## FinBERT labels per year",
        "",
        _md_table(finbert_labels),
        "",
        "## InflaBERT labels per year",
        "",
        _md_table(infla_labels),
        "",
        "## Months after S below min_articles_per_month",
        "",
        _md_table(flagged.loc[:, [col for col in ["month", "news_n_articles"] if col in flagged.columns]])
        if not flagged.empty
        else "_None._\n",
        "",
        "## Limitations",
        "",
        "- InflaBERT was trained on FNSPID **summaries** (<=100 words) from **2010 to 2019**. "
        "Scoring 2020 to 2023 is out of sample for InflaBERT.",
        "- Some 2010 to 2019 articles may have been in InflaBERT's training set, which can make "
        "its in-sample (Model 2 training period) fit look slightly better than it would be in real time.",
        "- We score **headlines** by default; InflaBERT saw summaries in training, so there is a "
        "small input mismatch. Optional variant: set `prefer_lsa_summary: true`.",
        "- FinBERT measures market mood, not inflation direction. Do not remap its labels by hand; "
        "Model 2 learns the sign.",
        "- InflaBERT's published `id2label` is `negative` / `neutral` / `positive`. This pipeline "
        "maps `positive` -> inflation-up and `negative` -> inflation-down. Confirm on Colab with "
        "headlines such as \"Gasoline prices soar\".",
        "- We do not label articles using realized CPI (that would leak the target into the features).",
        "- The 50-row sanity check is a human labeling exercise, not a training step. A teammate "
        "should mark `human_price_direction` as up / down / neither and report agreement in this file.",
        "",
        "## Open questions still for the team",
        "",
        "- Which FNSPID file (or the union) covers 1999 to 2023 best? `sample_rows` smoke tests cannot answer this.",
        "- Headline only, or summary when available?",
        "- Broad vs InflaBERT-narrow keyword list.",
        "- Isolated months below 30 articles after S: keep, drop, or impute?",
        "",
    ]
    dest.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Wrote diagnostics to %s", dest)
    return dest


def run_diagnostics(
    cfg: Model1Config,
    *,
    scored: pd.DataFrame,
    index: pd.DataFrame,
    full: pd.DataFrame,
    start: pd.Timestamp,
) -> dict[str, Path]:
    load_stats = _read_json(cfg.load_stats_path)
    cpi, cpi_error = fetch_cpi_mom(cfg)
    plot_path = cfg.reports_dir / "news_index_vs_cpi.png"
    plot_indexes(index, cpi, plot_path)
    sanity_path = write_sanity_sample(scored, cfg)
    notes_path = write_data_notes(cfg, start=start, full=full, load_stats=load_stats)
    report_path = write_diagnostics_report(
        cfg,
        scored=scored,
        index=index,
        full=full,
        start=start,
        cpi=cpi,
        cpi_error=cpi_error,
        plot_path=plot_path,
        sanity_path=sanity_path,
        load_stats=load_stats,
    )
    return {
        "plot": plot_path,
        "sanity": sanity_path,
        "data_notes": notes_path,
        "report": report_path,
    }
