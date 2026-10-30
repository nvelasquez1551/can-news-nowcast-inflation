"""Step 1–2: stream FNSPID CSVs, reduce columns, filter, write resumable chunks."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.model1.config import Model1Config
from src.model1.filter import compile_keyword_pattern, price_related_mask

LOGGER = logging.getLogger(__name__)

DATE_COL = "Date"
TITLE_COL = "Article_title"
SOURCE_COL = "source_file"
KEEP_AFTER_PARSE = [
    DATE_COL,
    TITLE_COL,
    "Stock_symbol",
    "Publisher",
    "Lsa_summary",
    SOURCE_COL,
]


def file_stem(relative_file: str) -> str:
    return Path(relative_file).stem


def source_dir(cfg: Model1Config, relative_file: str) -> Path:
    return cfg.filtered_dir / file_stem(relative_file)


def manifest_path(cfg: Model1Config, relative_file: str) -> Path:
    return source_dir(cfg, relative_file) / "manifest.json"


def parse_fnspid_dates(series: pd.Series) -> pd.Series:
    """Parse `YYYY-MM-DD HH:MM:SS UTC` as UTC; invalid values become NaT."""
    text = series.astype("string").str.replace(r"\s+UTC$", "", regex=True)
    return pd.to_datetime(text, utc=True, errors="coerce")


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _empty_manifest(relative_file: str, sample_rows: int | None) -> dict:
    return {
        "source_file": relative_file,
        "sample_rows": sample_rows,
        "complete": False,
        "chunks_written": 0,
        "raw_rows_read": 0,
        "n_raw": 0,
        "n_missing_date_or_title": 0,
        "n_duplicates": 0,
        "n_before_filter": 0,
        "n_after_filter": 0,
        "header_columns": [],
        "year_counts": {},
    }


def _usable_manifest(path: Path, sample_rows: int | None) -> dict | None:
    """Return a manifest only when sample_rows matches this run."""
    if not path.exists():
        return None
    payload = _read_json(path)
    if payload.get("sample_rows") != sample_rows:
        LOGGER.info(
            "Ignoring %s: sample_rows %s != current %s "
            "(a smoke-test manifest must not resume as a full run).",
            path,
            payload.get("sample_rows"),
            sample_rows,
        )
        return None
    return payload


def _assert_header(url: str, required: list[str]) -> list[str]:
    header = pd.read_csv(url, nrows=0, encoding="utf-8")
    columns = [str(col) for col in header.columns]
    LOGGER.info("Header for %s: %s", url, columns)
    missing = [col for col in required if col not in columns]
    if missing:
        raise AssertionError(
            f"Missing required columns {missing} in {url}. Header was {columns}."
        )
    extra = [col for col in columns if col not in required]
    if extra:
        LOGGER.info("Extra columns ignored via usecols: %s", extra)
    return columns


def _year_counts_from_dates(dates: pd.Series) -> dict[str, int]:
    years = pd.to_datetime(dates, utc=True, errors="coerce").dt.year.dropna().astype(int)
    return {str(year): int(count) for year, count in years.value_counts().items()}


def _add_year_counts(store: dict, before: pd.Series, after: pd.Series) -> None:
    for year, count in _year_counts_from_dates(before).items():
        bucket = store.setdefault(year, {"before": 0, "after": 0})
        bucket["before"] += int(count)
    for year, count in _year_counts_from_dates(after).items():
        bucket = store.setdefault(year, {"before": 0, "after": 0})
        bucket["after"] += int(count)


def _rebuild_seen(cfg: Model1Config) -> set[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    if not cfg.filtered_dir.exists():
        return seen
    for parquet_path in sorted(cfg.filtered_dir.glob("*/chunk_*.parquet")):
        chunk = pd.read_parquet(parquet_path, columns=[DATE_COL, TITLE_COL])
        if chunk.empty:
            continue
        dates = pd.to_datetime(chunk[DATE_COL], utc=True, errors="coerce")
        for date, title in zip(dates, chunk[TITLE_COL].astype(str), strict=False):
            if pd.isna(date):
                continue
            seen.add((date.date().isoformat(), title))
    LOGGER.info("Rebuilt %s seen (date, title) keys from existing filtered chunks.", len(seen))
    return seen


def _clean_chunk(
    frame: pd.DataFrame,
    *,
    relative_file: str,
    pattern,
    seen: set[tuple[str, str]],
) -> tuple[pd.DataFrame, dict, pd.Series, pd.Series]:
    stats = {
        "n_raw": int(len(frame)),
        "n_missing_date_or_title": 0,
        "n_duplicates": 0,
        "n_before_filter": 0,
        "n_after_filter": 0,
    }
    work = frame.copy()
    work[DATE_COL] = parse_fnspid_dates(work[DATE_COL])
    titles = work[TITLE_COL].astype("string").str.strip()
    valid = work[DATE_COL].notna() & titles.notna() & (titles != "")
    stats["n_missing_date_or_title"] = int((~valid).sum())
    work = work.loc[valid].copy()
    work[TITLE_COL] = titles.loc[valid]

    keys = [
        (ts.date().isoformat(), title)
        for ts, title in zip(work[DATE_COL], work[TITLE_COL], strict=False)
    ]
    keep_mask = []
    n_dup = 0
    for key in keys:
        if key in seen:
            n_dup += 1
            keep_mask.append(False)
        else:
            seen.add(key)
            keep_mask.append(True)
    stats["n_duplicates"] = n_dup
    work = work.loc[keep_mask].copy()
    stats["n_before_filter"] = int(len(work))
    before_dates = work[DATE_COL].copy() if not work.empty else pd.Series(
        dtype="datetime64[ns, UTC]"
    )

    if work.empty:
        empty = pd.DataFrame(columns=KEEP_AFTER_PARSE)
        stats["n_after_filter"] = 0
        return empty, stats, before_dates, pd.Series(dtype="datetime64[ns, UTC]")

    keep = price_related_mask(work, pattern)
    work = work.loc[keep].copy()
    work[SOURCE_COL] = relative_file
    stats["n_after_filter"] = int(len(work))
    after_dates = work[DATE_COL].copy() if not work.empty else pd.Series(
        dtype="datetime64[ns, UTC]"
    )
    return (
        work.loc[:, [col for col in KEEP_AFTER_PARSE if col in work.columns]],
        stats,
        before_dates,
        after_dates,
    )


def _merge_year_maps(parts: list[dict]) -> dict[str, dict[str, int]]:
    merged: dict[str, dict[str, int]] = defaultdict(lambda: {"before": 0, "after": 0})
    for part in parts:
        for year, counts in part.items():
            merged[str(year)]["before"] += int(counts.get("before", 0))
            merged[str(year)]["after"] += int(counts.get("after", 0))
    return dict(sorted(merged.items()))


def stream_and_filter(cfg: Model1Config) -> pd.DataFrame:
    """Stream each FNSPID file, write filtered parquet chunks, concat the union."""
    cfg.filtered_dir.mkdir(parents=True, exist_ok=True)
    pattern = compile_keyword_pattern(cfg.keywords)
    seen = _rebuild_seen(cfg)
    file_stats: list[dict] = []

    for relative_file in cfg.dataset_files:
        stats = _process_source_file(cfg, relative_file, pattern, seen)
        file_stats.append(stats)

    filtered = _concat_filtered_chunks(cfg)
    year_counts = _merge_year_maps([item.get("year_counts", {}) for item in file_stats])
    _write_year_counts(cfg.year_counts_path, year_counts)
    load_stats = {
        "sample_rows": cfg.sample_rows,
        "files": file_stats,
        "n_filtered_rows": int(len(filtered)),
        "year_counts": year_counts,
    }
    _write_json(cfg.load_stats_path, load_stats)
    LOGGER.info("Filtered union: %s rows -> %s", len(filtered), cfg.news_filtered_path)
    return filtered


def _process_source_file(
    cfg: Model1Config,
    relative_file: str,
    pattern,
    seen: set[tuple[str, str]],
) -> dict:
    out_dir = source_dir(cfg, relative_file)
    out_dir.mkdir(parents=True, exist_ok=True)
    man_path = manifest_path(cfg, relative_file)
    existing = _usable_manifest(man_path, cfg.sample_rows)
    if existing is not None and existing.get("complete"):
        LOGGER.info(
            "Skipping completed source %s (sample_rows=%s).",
            relative_file,
            cfg.sample_rows,
        )
        return existing

    url = cfg.hf_url(relative_file)
    header_columns = _assert_header(url, cfg.columns_keep)
    manifest = existing or _empty_manifest(relative_file, cfg.sample_rows)
    manifest["header_columns"] = header_columns

    raw_rows_read = int(manifest.get("raw_rows_read", 0))
    chunks_written = int(manifest.get("chunks_written", 0))
    remaining = None if cfg.sample_rows is None else max(cfg.sample_rows - raw_rows_read, 0)
    if remaining == 0:
        manifest["complete"] = True
        _write_json(man_path, manifest)
        return manifest

    # Callable skip is O(1) memory; a range of millions of row numbers is not.
    skiprows = None
    if raw_rows_read:
        n_skip = raw_rows_read
        skiprows = lambda idx, n=n_skip: idx != 0 and idx <= n
    LOGGER.info(
        "Streaming %s (skip_raw_rows=%s, remaining_cap=%s).",
        url,
        raw_rows_read,
        remaining,
    )
    reader = pd.read_csv(
        url,
        usecols=cfg.columns_keep,
        dtype=str,
        chunksize=cfg.chunksize,
        skiprows=skiprows,
        encoding="utf-8",
    )

    year_counts = dict(manifest.get("year_counts") or {})
    for chunk in reader:
        if remaining is not None:
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()
            remaining -= len(chunk)

        n_raw = int(len(chunk))
        cleaned, stats, before_dates, after_dates = _clean_chunk(
            chunk, relative_file=relative_file, pattern=pattern, seen=seen
        )
        _add_year_counts(year_counts, before_dates, after_dates)

        chunk_path = out_dir / f"chunk_{chunks_written:05d}.parquet"
        if cleaned.empty:
            pd.DataFrame(columns=KEEP_AFTER_PARSE).to_parquet(chunk_path, index=False)
        else:
            cleaned.to_parquet(chunk_path, index=False)

        chunks_written += 1
        raw_rows_read += n_raw
        manifest["n_raw"] = int(manifest.get("n_raw", 0)) + stats["n_raw"]
        manifest["n_missing_date_or_title"] = int(
            manifest.get("n_missing_date_or_title", 0)
        ) + stats["n_missing_date_or_title"]
        manifest["n_duplicates"] = int(manifest.get("n_duplicates", 0)) + stats["n_duplicates"]
        manifest["n_before_filter"] = int(manifest.get("n_before_filter", 0)) + stats[
            "n_before_filter"
        ]
        manifest["n_after_filter"] = int(manifest.get("n_after_filter", 0)) + stats[
            "n_after_filter"
        ]
        manifest["chunks_written"] = chunks_written
        manifest["raw_rows_read"] = raw_rows_read
        manifest["year_counts"] = year_counts
        _write_json(man_path, manifest)
        LOGGER.info(
            "%s chunk %s: raw=%s kept=%s (running kept=%s)",
            relative_file,
            chunks_written - 1,
            n_raw,
            stats["n_after_filter"],
            manifest["n_after_filter"],
        )
        if remaining is not None and remaining <= 0:
            break

    manifest["complete"] = True
    manifest["year_counts"] = year_counts
    _write_json(man_path, manifest)
    return manifest


def _write_year_counts(path: Path, year_counts: dict[str, dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "year": int(year),
            "n_before_filter": counts.get("before", 0),
            "n_after_filter": counts.get("after", 0),
        }
        for year, counts in sorted(year_counts.items(), key=lambda item: int(item[0]))
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _concat_filtered_chunks(cfg: Model1Config) -> pd.DataFrame:
    paths = sorted(cfg.filtered_dir.glob("*/chunk_*.parquet"))
    if not paths:
        filtered = pd.DataFrame(columns=KEEP_AFTER_PARSE)
    else:
        frames = [pd.read_parquet(path) for path in paths]
        filtered = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=KEEP_AFTER_PARSE)
        )
    cfg.interim_dir.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(cfg.news_filtered_path, index=False)
    return filtered
