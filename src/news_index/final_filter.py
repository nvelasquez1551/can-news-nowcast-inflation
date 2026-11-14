"""Filtro final (offline) sobre Alpaca y FNSPID con keywords_final.

Reglas para AMBAS fuentes:
- Match solo en el TÍTULO, palabra completa, sin mayúsculas (keywords_final).
- Se descartan títulos que mencionan exclude_entities o que coinciden con
  exclude_title_patterns.
- Dedupe por (día UTC, título).

Las entradas ya están prefiltradas con keywords_broad; cada término de
keywords_final contiene un término broad, así que no se pierde nada.

Uso:
    python -m src.news_index.final_filter
"""

import re

import pandas as pd
import yaml

from src.news_index.download_alpaca import ROOT, matched_terms

CONFIG_PATH = ROOT / "config" / "news_index.yaml"
ALPACA_DIR = ROOT / "data" / "raw" / "alpaca"
FNSPID_MATCHED = ROOT / "data" / "raw" / "fnspid" / "matched_external.parquet"
INTERIM = ROOT / "data" / "interim"
OUT_ALPACA = INTERIM / "final_filter_alpaca.parquet"
OUT_FNSPID = INTERIM / "final_filter_fnspid.parquet"
OUT_COUNTS = INTERIM / "final_filter_monthly_counts.csv"

ALPACA_RANGE = ("2015-01", "2026-10")
MIN_ARTICLES = 30


def load_config():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    terms = {t: re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE)
             for t in cfg["keywords_final"]}
    excludes = {}
    for e in cfg["exclude_entities"]:
        excludes[f"entity: {e}"] = re.compile(rf"\b{re.escape(e)}\b",
                                              re.IGNORECASE)
    for p in cfg["exclude_title_patterns"]:
        excludes[f"pattern: {p}"] = re.compile(p, re.IGNORECASE)
    return terms, excludes


def apply_filter(df, terms, excludes, name):
    """df con columnas date (UTC), title, domain. Devuelve (filtrado,
    conteos por regla de exclusión)."""
    df = df[df["title"].fillna("").str.strip() != ""].copy()
    n_in = len(df)
    # 1. Match en el título con keywords_final
    df["matched_terms"] = df["title"].map(lambda t: matched_terms(t, terms))
    df = df[df["matched_terms"].map(bool)]
    n_kw = len(df)
    # 2. Exclusiones (cada regla contada por separado; pueden solaparse)
    removed = {}
    drop = pd.Series(False, index=df.index)
    for rule, rx in excludes.items():
        hit = df["title"].str.contains(rx)
        removed[rule] = int(hit.sum())
        drop |= hit
    df = df[~drop]
    n_excl = int(drop.sum())
    # 3. Dedupe por (día UTC, título)
    day = df["date"].dt.floor("D")
    dup = pd.DataFrame({"d": day, "t": df["title"]}).duplicated()
    df = df[~dup.to_numpy()]
    print(f"\n##### {name} #####")
    print(f"input titles: {n_in:,} -> keyword match: {n_kw:,} -> after "
          f"excludes: {n_kw - n_excl:,} (-{n_excl:,}) -> after dedupe: "
          f"{len(df):,} (-{int(dup.sum()):,})")
    df["month"] = df["date"].dt.strftime("%Y-%m")
    return df[["date", "title", "domain", "matched_terms", "month"]], removed


def monthly_counts(df, first, last):
    months = pd.period_range(first, last, freq="M").strftime("%Y-%m")
    return df["month"].value_counts().reindex(months, fill_value=0) \
        .rename("n_matched").rename_axis("month").reset_index()


def report(df, counts, removed, terms):
    print("\nTitles removed by each exclude rule (on keyword-matched titles, "
          "before dedupe; rules may overlap):")
    for rule, n in removed.items():
        print(f"  {n:7,}  {rule}")

    c = counts.copy()
    c["year"] = c["month"].str[:4]
    t = c.groupby("year")["n_matched"].agg(
        months="size", total="sum", min="min", median="median", max="max")
    print("\nBy year (months in range, zero-filled):")
    print(t.to_string())

    low = c[c["n_matched"] < MIN_ARTICLES]
    print(f"\nMonths with n < {MIN_ARTICLES}: {len(low)}")
    if len(low):
        print("  " + ", ".join(f"{m} ({n})" for m, n in
                               zip(low["month"], low["n_matched"])))

    lg = df[["month", "matched_terms"]].explode("matched_terms")
    lg["year"] = lg["month"].str[:4]
    overall = lg["matched_terms"].value_counts().reindex(terms, fill_value=0)
    print("\nTerm share of matched titles, overall "
          "(titles can match several terms):")
    for term, n in overall.sort_values(ascending=False).items():
        print(f"  {term:22s} {n:7,}  {n / len(df):6.1%}")
    n_year = df["month"].str[:4].value_counts()
    by_year = lg.groupby(["matched_terms", "year"]).size().unstack(
        fill_value=0).reindex(terms, fill_value=0)
    print("\nTerm share by year (%):")
    print((100 * by_year / n_year).round(1).to_string())


def main():
    terms, excludes = load_config()
    pd.set_option("display.width", 250)

    # --- Alpaca: solo headline, mes por created_at, rango 2015-01..2026-10
    al = pd.concat([pd.read_parquet(p, columns=["created_at", "headline"])
                    for p in sorted(ALPACA_DIR.glob("*.parquet"))],
                   ignore_index=True)
    al = pd.DataFrame({"date": al["created_at"], "title": al["headline"],
                       "domain": "alpaca"})
    months = al["date"].dt.strftime("%Y-%m")
    in_range = months.between(*ALPACA_RANGE)
    print(f"Alpaca: dropped {int((~in_range).sum())} rows with created_at "
          f"outside {ALPACA_RANGE[0]}..{ALPACA_RANGE[1]}")
    al_f, al_removed = apply_filter(al[in_range], terms, excludes, "ALPACA")
    al_counts = monthly_counts(al_f, *ALPACA_RANGE)
    report(al_f, al_counts, al_removed, list(terms))

    # --- FNSPID
    fn = pd.read_parquet(FNSPID_MATCHED, columns=["date", "title", "domain"])
    fn_f, fn_removed = apply_filter(fn, terms, excludes, "FNSPID")
    fn_counts = monthly_counts(fn_f, fn_f["month"].min(),
                               fn_f["month"].max())
    report(fn_f, fn_counts, fn_removed, list(terms))

    # --- Guardar
    al_f.drop(columns="month").to_parquet(OUT_ALPACA, index=False)
    fn_f.drop(columns="month").to_parquet(OUT_FNSPID, index=False)
    pd.concat([al_counts.assign(source="alpaca"),
               fn_counts.assign(source="fnspid")])[
        ["source", "month", "n_matched"]].to_csv(OUT_COUNTS, index=False)
    print(f"\nsaved {OUT_ALPACA}\nsaved {OUT_FNSPID}\nsaved {OUT_COUNTS}")


if __name__ == "__main__":
    main()
