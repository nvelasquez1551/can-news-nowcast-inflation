"""Descarga reanudable de noticias de Alpaca (Benzinga), mes a mes.

Solo guarda los artículos que coinciden con el léxico BROAD.

Uso:
    python -m src.news_index.download_alpaca                  # 2015-01 .. 2026-10
    python -m src.news_index.download_alpaca --only 2015-06
    python -m src.news_index.download_alpaca --start 2018-01 --end 2018-12
"""

import argparse
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "news_index.yaml"
RAW_DIR = ROOT / "data" / "raw" / "alpaca"
COUNTS_CSV = ROOT / "data" / "interim" / "alpaca_counts.csv"
LOG_PATH = ROOT / "logs" / "download_alpaca.log"

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
DEFAULT_START = "2015-01"
DEFAULT_END = "2026-10"
MIN_SECONDS_BETWEEN_CALLS = 60 / 170   # margen bajo el límite de 180/min
MAX_RETRIES = 5
TIMEOUT = 60

log = logging.getLogger("download_alpaca")


# --- Seguridad y setup ---------------------------------------------------

def check_gitignore():
    """Aborta si data/ o logs/ no están ignorados por git."""
    missing = []
    for folder in ["data/", "logs/"]:
        res = subprocess.run(["git", "check-ignore", "-q", folder], cwd=ROOT)
        if res.returncode != 0:
            missing.append(folder)
    if missing:
        sys.exit(f"ABORT: {', '.join(missing)} not in .gitignore. "
                 "Add it before downloading.")


def setup_logging():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    log.setLevel(logging.INFO)
    for handler in [logging.FileHandler(LOG_PATH),
                    logging.StreamHandler(sys.stdout)]:
        handler.setFormatter(fmt)
        log.addHandler(handler)


def load_headers():
    """Headers con las claves (nunca se imprimen ni se loguean)."""
    load_dotenv(ROOT / ".env")
    key = os.getenv("ALPACA_API_KEY_ID")
    secret = os.getenv("ALPACA_API_SECRET_KEY")
    if not key or not secret:
        sys.exit("ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY missing in .env")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


# --- Léxico --------------------------------------------------------------

def load_term_patterns():
    """Una regex por término: palabra completa, sin mayúsculas."""
    with open(CONFIG_PATH) as f:
        terms = yaml.safe_load(f)["keywords_broad"]
    return {t: re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE)
            for t in terms}


def matched_terms(text, patterns):
    """Lista de términos que aparecen en el texto."""
    return [t for t, p in patterns.items() if p.search(text)]


# --- Meses ---------------------------------------------------------------

def month_list(start, end):
    """'2015-01', '2015-03' -> ['2015-01', '2015-02', '2015-03']."""
    y, m = map(int, start.split("-"))
    y_end, m_end = map(int, end.split("-"))
    out = []
    while (y, m) <= (y_end, m_end):
        out.append(f"{y}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def month_bounds(month):
    """Inicio y fin (exclusivo) en RFC-3339."""
    y, m = map(int, month.split("-"))
    y2, m2 = (y + 1, 1) if m == 12 else (y, m + 1)
    return f"{y}-{m:02d}-01T00:00:00Z", f"{y2}-{m2:02d}-01T00:00:00Z"


# --- API -----------------------------------------------------------------

class Client:
    def __init__(self, headers):
        self.session = requests.Session()
        self.session.headers.update(headers)
        self.last_call = 0.0

    def get(self, params):
        """Una llamada con control de ritmo y reintentos."""
        n_errors = 0
        while True:
            wait = MIN_SECONDS_BETWEEN_CALLS - (time.time() - self.last_call)
            if wait > 0:
                time.sleep(wait)
            self.last_call = time.time()
            try:
                r = self.session.get(NEWS_URL, params=params, timeout=TIMEOUT)
            except (requests.Timeout, requests.ConnectionError) as err:
                status = type(err).__name__
            else:
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 429:
                    log.warning("HTTP 429, sleeping 60s")
                    time.sleep(60)
                    continue   # 429 no cuenta como error
                if r.status_code < 500:
                    # 401/403/400...: no tiene sentido reintentar
                    sys.exit(f"HTTP {r.status_code}: {r.text[:200]}")
                status = f"HTTP {r.status_code}"
            # 5xx o timeout: backoff exponencial, máximo 5 reintentos
            n_errors += 1
            if n_errors > MAX_RETRIES:
                raise RuntimeError(f"{status} after {MAX_RETRIES} retries")
            backoff = 5 * 2 ** (n_errors - 1)
            log.warning(f"{status}, retry {n_errors}/{MAX_RETRIES} "
                        f"in {backoff}s")
            time.sleep(backoff)


def download_month(client, month, patterns):
    """Recorre todas las páginas del mes. Devuelve (DataFrame de
    artículos que coinciden, n_calls, n_total)."""
    start, end = month_bounds(month)
    params = {"start": start, "end": end, "sort": "asc", "limit": 50,
              "include_content": "false"}
    seen = set()
    rows = []
    n_calls = 0
    while True:
        j = client.get(params)
        n_calls += 1
        for a in j.get("news", []):
            # Dedupe por id y fin exclusivo (por si la API incluye el borde)
            if a["id"] in seen or a.get("created_at", "") >= end:
                continue
            seen.add(a["id"])
            headline = a.get("headline") or ""
            summary = a.get("summary") or ""
            terms = matched_terms(headline + "\n" + summary, patterns)
            if terms:
                rows.append({"id": a["id"], "created_at": a["created_at"],
                             "headline": headline, "summary": summary,
                             "matched_terms": terms})
        token = j.get("next_page_token")
        if not token:
            break
        params["page_token"] = token
    df = pd.DataFrame(rows, columns=["id", "created_at", "headline",
                                     "summary", "matched_terms"])
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    return df, n_calls, len(seen)


# --- Guardado ------------------------------------------------------------

def save_month(df, month):
    """Escribe a un temporal y renombra: nunca queda un mes a medias."""
    final = RAW_DIR / f"{month}.parquet"
    tmp = RAW_DIR / f"{month}.parquet.tmp"
    df.to_parquet(tmp, index=False)
    tmp.replace(final)


def append_counts(month, n_calls, n_total, n_matched):
    """Agrega la fila del mes (reemplaza una fila previa del mismo mes)."""
    row = pd.DataFrame([{"month": month, "n_calls": n_calls,
                         "n_total": n_total, "n_matched": n_matched}])
    if COUNTS_CSV.exists():
        old = pd.read_csv(COUNTS_CSV, dtype={"month": str})
        row = pd.concat([old[old["month"] != month], row])
    row.sort_values("month").to_csv(COUNTS_CSV, index=False)


def fmt_seconds(s):
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m{s % 60:02d}s"


# --- Main ----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--start", default=DEFAULT_START, help="YYYY-MM")
    p.add_argument("--end", default=DEFAULT_END, help="YYYY-MM, inclusive")
    p.add_argument("--only", help="single month YYYY-MM")
    return p.parse_args()


def main():
    args = parse_args()
    check_gitignore()
    setup_logging()
    client = Client(load_headers())
    patterns = load_term_patterns()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    COUNTS_CSV.parent.mkdir(parents=True, exist_ok=True)

    months = [args.only] if args.only else month_list(args.start, args.end)
    todo = [m for m in months if not (RAW_DIR / f"{m}.parquet").exists()]
    log.info(f"{len(months)} months requested, {len(months) - len(todo)} "
             f"already done, {len(todo)} to download "
             f"({len(patterns)} terms)")

    t0 = time.time()
    for i, month in enumerate(todo, start=1):
        df, n_calls, n_total = download_month(client, month, patterns)
        save_month(df, month)
        append_counts(month, n_calls, n_total, len(df))
        elapsed = time.time() - t0
        eta = elapsed / i * (len(todo) - i)
        log.info(f"[{i}/{len(todo)}] {month}: calls={n_calls} "
                 f"total={n_total} matched={len(df)} | "
                 f"elapsed {fmt_seconds(elapsed)} ETA {fmt_seconds(eta)}")
    log.info("done")


if __name__ == "__main__":
    main()
