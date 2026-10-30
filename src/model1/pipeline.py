"""CLI: stream/filter -> score -> monthly index -> diagnostics.

Usage:
    python -m src.model1.pipeline --config config/model1.yaml
    python -m src.model1.pipeline --data-root /content/drive/MyDrive/can-news-nowcast-inflation/data \\
        --reports-dir /content/drive/MyDrive/can-news-nowcast-inflation/reports --sample-rows 50000
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from src.model1.aggregate import INDEX_COLUMNS, build_news_index
from src.model1.config import load_config
from src.model1.diagnostics import run_diagnostics
from src.model1.load import stream_and_filter
from src.model1.score import score_filtered_chunks

LOGGER = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Model 1 monthly news indexes.")
    parser.add_argument("--config", default="config/model1.yaml", help="Path to model1.yaml")
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override data/ so Colab can write interim+processed to Google Drive.",
    )
    parser.add_argument(
        "--reports-dir",
        default=None,
        help="Override reports/ (and docs/data_notes.md sibling) for Drive output.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Cap raw rows per FNSPID file (smoke test). Omit for the full run.",
    )
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    cfg = load_config(
        args.config,
        data_root=args.data_root,
        reports_dir=args.reports_dir,
        sample_rows=args.sample_rows,
    )
    LOGGER.info(
        "Model 1 start: sample_rows=%s cutoff_day=%s data_root=%s",
        cfg.sample_rows,
        cfg.cutoff_day,
        cfg.data_root,
    )

    stream_and_filter(cfg)
    scored = score_filtered_chunks(cfg)
    index, full, start = build_news_index(scored, cfg)
    run_diagnostics(cfg, scored=scored, index=index, full=full, start=start)

    year_counts = (
        pd.read_csv(cfg.year_counts_path) if cfg.year_counts_path.exists() else pd.DataFrame()
    )
    flagged = full.loc[full["below_min_flag"]] if "below_min_flag" in full.columns else full.iloc[0:0]
    print("\n=== Model 1 summary ===")
    print(f"Start month S: {pd.Timestamp(start).strftime('%Y-%m-%d')}")
    print(f"Index path: {cfg.index_csv_path}")
    print(f"Index rows: {len(index)}")
    print(f"Columns: {INDEX_COLUMNS}")
    print(f"Months after S below {cfg.min_articles_per_month} articles: {len(flagged)}")
    if not year_counts.empty:
        print("\nArticles per year (before -> after keyword filter):")
        for row in year_counts.itertuples(index=False):
            print(f"  {int(row.year)}: {int(row.n_before_filter)} -> {int(row.n_after_filter)}")
    if len(flagged):
        print("\nFlagged months:")
        for ts, n in zip(flagged["month"], flagged["news_n_articles"], strict=False):
            print(f"  {pd.Timestamp(ts).strftime('%Y-%m-%d')}: n={int(n)}")
    return 0


def main() -> None:
    sys.exit(run())


if __name__ == "__main__":
    main()
