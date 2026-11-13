"""Escaneo de FNSPID Stock_news/All_external.csv (solo títulos).

Descarga el archivo (reanudable), deduplica, aplica el mismo léxico que
download_alpaca.py sobre el TÍTULO y guarda conteos y artículos que coinciden.

Uso:
    python -m src.news_index.scan_fnspid
"""

import shutil
import subprocess
import sys
import time
from urllib.parse import urlparse

import pandas as pd
from huggingface_hub import hf_hub_download

from src.news_index.download_alpaca import (ROOT, load_term_patterns,
                                            matched_terms)

REPO_ID = "Zihan1004/FNSPID"
# Misma revisión fijada en config/model1.yaml (reproducibilidad)
REVISION = "bf9189c41527198897d1af3e17b1a0095279fc45"
FILENAME = "Stock_news/All_external.csv"
MIN_FREE_GB = 15
CHUNKSIZE = 500_000
USECOLS = ["Date", "Article_title", "Stock_symbol", "Url", "Publisher"]

FNSPID_DIR = ROOT / "data" / "raw" / "fnspid"
MATCHED_PARQUET = FNSPID_DIR / "matched_external.parquet"
COUNTS_CSV = ROOT / "data" / "interim" / "fnspid_counts.csv"
DOMAIN_CSV = ROOT / "data" / "interim" / "fnspid_domain_by_year.csv"


def check_safety():
    """Espacio libre >= 15 GB y data/ ignorado por git."""
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    print(f"free disk: {free_gb:.0f} GB")
    if free_gb < MIN_FREE_GB:
        sys.exit(f"ABORT: less than {MIN_FREE_GB} GB free")
    res = subprocess.run(["git", "check-ignore", "-q", "data/"], cwd=ROOT)
    if res.returncode != 0:
        sys.exit("ABORT: data/ not in .gitignore")


def download():
    """hf_hub_download es reanudable y muestra barra de progreso."""
    FNSPID_DIR.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset",
                           filename=FILENAME, revision=REVISION,
                           local_dir=FNSPID_DIR)
    print(f"file: {path}")
    return path


def domain_of(url):
    """'https://www.benzinga.com/x' -> 'benzinga.com'."""
    if not isinstance(url, str):
        return ""
    netloc = urlparse(url).netloc.lower()
    return netloc[4:] if netloc.startswith("www.") else netloc


def main():
    check_safety()
    path = download()

    patterns = load_term_patterns()
    # Prefiltro con la unión de los mismos 25 patrones (vectorizado);
    # matched_terms() se aplica solo a los títulos que pasan.
    any_term = "|".join(f"(?:{p.pattern})" for p in patterns.values())

    seen = set()          # hashes de 64 bits de (Date, Article_title)
    counts = {}           # month -> [n_unique, n_matched]
    domain_year = {}      # (year, domain) -> n_unique
    matched_frames = []
    n_rows = n_null_title = n_bad_date = 0
    t0 = time.time()

    reader = pd.read_csv(path, usecols=USECOLS, chunksize=CHUNKSIZE,
                         dtype=str)
    for i, chunk in enumerate(reader, start=1):
        n_rows += len(chunk)
        # Quitar títulos nulos o vacíos
        has_title = chunk["Article_title"].fillna("").str.strip() != ""
        n_null_title += int((~has_title).sum())
        chunk = chunk[has_title]
        # Deduplicación global por (Date, Article_title): una fila por ticker
        h = pd.util.hash_pandas_object(chunk[["Date", "Article_title"]],
                                       index=False).to_numpy()
        keep = []
        for x in h:
            if x in seen:
                keep.append(False)
            else:
                seen.add(x)
                keep.append(True)
        chunk = chunk[keep].copy()
        # Fecha en UTC -> mes
        chunk["date"] = pd.to_datetime(chunk["Date"], utc=True,
                                       errors="coerce")
        n_bad_date += int(chunk["date"].isna().sum())
        chunk = chunk[chunk["date"].notna()]
        chunk["month"] = chunk["date"].dt.strftime("%Y-%m")
        chunk["domain"] = chunk["Url"].map(domain_of)
        # Coincidencias en el TÍTULO solamente
        hit = chunk["Article_title"].str.contains(any_term, case=False,
                                                  regex=True)
        m = chunk[hit].copy()
        m["matched_terms"] = m["Article_title"].map(
            lambda t: matched_terms(t, patterns))
        matched_frames.append(pd.DataFrame({
            "date": m["date"], "title": m["Article_title"],
            "domain": m["domain"], "publisher": m["Publisher"],
            "matched_terms": m["matched_terms"]}))
        # Conteos
        for month, n in chunk["month"].value_counts().items():
            counts.setdefault(month, [0, 0])[0] += n
        for month, n in m["month"].value_counts().items():
            counts.setdefault(month, [0, 0])[1] += n
        yd = chunk.groupby([chunk["date"].dt.year, "domain"]).size()
        for key, n in yd.items():
            domain_year[key] = domain_year.get(key, 0) + n
        print(f"chunk {i}: rows read {n_rows:,}, unique kept {len(seen):,}, "
              f"matched so far {sum(len(f) for f in matched_frames):,} "
              f"({time.time() - t0:.0f}s)", flush=True)

    matched = pd.concat(matched_frames, ignore_index=True)
    matched.to_parquet(MATCHED_PARQUET, index=False)
    pd.DataFrame([{"month": k, "n_unique": v[0], "n_matched": v[1]}
                  for k, v in sorted(counts.items())]
                 ).to_csv(COUNTS_CSV, index=False)
    pd.DataFrame([{"year": y, "domain": d, "n_unique": n}
                  for (y, d), n in sorted(domain_year.items())]
                 ).to_csv(DOMAIN_CSV, index=False)
    print(f"\nrows read: {n_rows:,}  null/empty titles dropped: "
          f"{n_null_title:,}  unique (Date, title): {len(seen):,}  "
          f"unparseable dates dropped: {n_bad_date:,}")
    print(f"saved {MATCHED_PARQUET} ({len(matched):,} rows)")
    print(f"saved {COUNTS_CSV}")
    print(f"saved {DOMAIN_CSV}")


if __name__ == "__main__":
    main()
