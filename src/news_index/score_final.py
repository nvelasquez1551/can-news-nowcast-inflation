"""Puntúa con InflaBERT los titulares del filtro final (Alpaca y FNSPID).

- Entrada: data/interim/final_filter_<source>.parquet (date, title, ...).
- Solo el TITULAR; FinBERT no se usa.
- Modelo y revisión desde config/model1.yaml (models.inflabert).
- Chunks reanudables en data/processed/news_scored_<source>_chunks/ y un
  archivo final data/processed/news_scored_<source>.parquet.

Uso:
    python -m src.news_index.score_final --source all          # corrida completa (Colab/GPU)
    python -m src.news_index.score_final --source alpaca --limit 500   # smoke test
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from src.model1.config import load_config
from src.model1.score import load_both_scorers, score_frame

ROOT = Path(__file__).resolve().parents[2]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
SOURCES = ["alpaca", "fnspid"]

log = logging.getLogger("score_final")


def out_paths(source, limit):
    """Rutas de salida; las corridas con --limit van a archivos aparte."""
    tag = source if limit is None else f"{source}_limit{limit}"
    return (PROCESSED / f"news_scored_{tag}_chunks",
            PROCESSED / f"news_scored_{tag}.parquet")


def load_input(source, limit):
    df = pd.read_parquet(INTERIM / f"final_filter_{source}.parquet")
    df = df.rename(columns={"date": "Date", "title": "Article_title"})
    df = df.sort_values(["Date", "Article_title"], kind="stable") \
        .reset_index(drop=True)
    if limit is not None:
        df = df.head(limit)
    return df


def check_manifest(chunk_dir, n_rows, chunk_size):
    """Evita mezclar chunks de una entrada distinta al reanudar."""
    path = chunk_dir / "manifest.json"
    meta = {"n_rows": n_rows, "chunk_size": chunk_size}
    if path.exists():
        old = json.loads(path.read_text())
        if old != meta:
            sys.exit(f"ABORT: {chunk_dir} was written for {old}, now {meta}. "
                     "Delete that folder to start over.")
    else:
        chunk_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta))


def score_source(source, cfg, scorers_box, limit, chunk_size):
    df = load_input(source, limit)
    chunk_dir, final = out_paths(source, limit)
    check_manifest(chunk_dir, len(df), chunk_size)
    starts = list(range(0, len(df), chunk_size))
    todo = [s for s in starts
            if not (chunk_dir / f"chunk_{s // chunk_size:05d}.parquet").exists()]
    log.info(f"{source}: {len(df):,} headlines, {len(starts)} chunks, "
             f"{len(todo)} to score")

    t0 = time.time()
    for i, s in enumerate(todo, start=1):
        if scorers_box.get("scorers") is None:
            t_load = time.time()
            scorers_box["scorers"] = load_both_scorers(cfg)
            log.info(f"model load: {time.time() - t_load:.1f}s "
                     f"(scorers: {list(scorers_box['scorers'])})")
        t_chunk = time.time()
        part = df.iloc[s:s + chunk_size]
        scored = score_frame(part, scorers_box["scorers"], cfg)
        path = chunk_dir / f"chunk_{s // chunk_size:05d}.parquet"
        tmp = path.with_suffix(".tmp")
        scored.to_parquet(tmp, index=False)
        tmp.replace(path)
        dt = time.time() - t_chunk
        elapsed = time.time() - t0
        eta = elapsed / i * (len(todo) - i)
        log.info(f"{source} chunk {i}/{len(todo)}: {len(part):,} rows in "
                 f"{dt:.1f}s ({len(part) / dt:.1f} rows/s) | ETA {eta / 60:.1f} min")

    chunks = sorted(chunk_dir.glob("chunk_*.parquet"))
    out = pd.concat([pd.read_parquet(p) for p in chunks], ignore_index=True)
    out.to_parquet(final, index=False)
    log.info(f"wrote {final} ({len(out):,} rows)")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--source", choices=SOURCES + ["all"], default="all")
    p.add_argument("--limit", type=int, help="score only the first N headlines")
    p.add_argument("--chunk-size", type=int, default=5000)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    # Solo InflaBERT, solo titulares
    cfg.finbert_enabled = False
    cfg.prefer_lsa_summary = False
    log.info(f"InflaBERT {cfg.inflabert.name} @ {cfg.inflabert.revision}, "
             f"batch_size={cfg.batch_size}, max_length={cfg.max_length}")

    scorers_box = {}
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for source in (SOURCES if args.source == "all" else [args.source]):
        t0 = time.time()
        out = score_source(source, cfg, scorers_box, args.limit,
                           args.chunk_size)
        log.info(f"{source}: done in {time.time() - t0:.1f}s; "
                 f"infla_role counts: {out['infla_role'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
