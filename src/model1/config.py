"""Load config/model1.yaml and apply CLI path / sample_rows overrides."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "model1.yaml"


@dataclass
class ModelSpec:
    key: str
    name: str
    revision: str
    role_aliases: dict[str, list[str]]
    score_positive_role: str
    score_negative_role: str


@dataclass
class Model1Config:
    raw: dict[str, Any]
    repo_root: Path
    dataset_name: str
    dataset_revision: str
    dataset_files: list[str]
    columns_keep: list[str]
    finbert: ModelSpec
    inflabert: ModelSpec
    cutoff_day: int
    min_articles_per_month: int
    consecutive_months: int
    batch_size: int
    max_length: int
    chunksize: int
    sample_rows: int | None
    use_narrow_lexicon: bool
    prefer_lsa_summary: bool
    sanity_sample_size: int
    sanity_sample_seed: int
    keywords: list[str]
    keywords_broad: list[str]
    keywords_narrow: list[str]
    data_root: Path
    interim_dir: Path
    processed_dir: Path
    reports_dir: Path
    data_notes_path: Path
    cpi_url: str
    cpi_date_col: str
    cpi_value_col: str

    @property
    def sample_key(self) -> str:
        return "full" if self.sample_rows is None else str(self.sample_rows)

    @property
    def filtered_dir(self) -> Path:
        return self.interim_dir / "filtered" / f"sample_{self.sample_key}"

    @property
    def scored_dir(self) -> Path:
        return self.processed_dir / "scored" / f"sample_{self.sample_key}"

    @property
    def news_filtered_path(self) -> Path:
        return self.interim_dir / "news_filtered.parquet"

    @property
    def news_scored_path(self) -> Path:
        return self.processed_dir / "news_scored.parquet"

    @property
    def year_counts_path(self) -> Path:
        return self.interim_dir / "year_counts.csv"

    @property
    def load_stats_path(self) -> Path:
        return self.interim_dir / "load_stats.json"

    @property
    def index_csv_path(self) -> Path:
        return self.processed_dir / "news_index_monthly.csv"

    @property
    def index_parquet_path(self) -> Path:
        return self.processed_dir / "news_index_monthly.parquet"

    @property
    def index_full_path(self) -> Path:
        return self.processed_dir / "news_index_monthly_full.csv"

    @property
    def start_month_path(self) -> Path:
        return self.processed_dir / "start_month.txt"

    def hf_url(self, relative_file: str) -> str:
        return (
            f"hf://datasets/{self.dataset_name}@{self.dataset_revision}/{relative_file}"
        )


def _as_optional_int(value: Any) -> int | None:
    if value is None or value == "null":
        return None
    return int(value)


def _model_spec(key: str, payload: dict[str, Any]) -> ModelSpec:
    aliases = {
        str(role): [str(alias) for alias in alias_list]
        for role, alias_list in payload["role_aliases"].items()
    }
    return ModelSpec(
        key=key,
        name=str(payload["name"]),
        revision=str(payload["revision"]),
        role_aliases=aliases,
        score_positive_role=str(payload["score_positive_role"]),
        score_negative_role=str(payload["score_negative_role"]),
    )


def _resolve(path_value: str | Path, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def load_config(
    path: str | Path | None = None,
    *,
    data_root: str | Path | None = None,
    reports_dir: str | Path | None = None,
    sample_rows: int | None = None,
    repo_root: str | Path | None = None,
) -> Model1Config:
    repo = Path(repo_root).resolve() if repo_root is not None else REPO_ROOT
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = (repo / config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    paths = raw["paths"]
    root = _resolve(data_root if data_root is not None else paths["data_root"], repo)
    if data_root is not None:
        interim = root / "interim"
        processed = root / "processed"
    else:
        interim = _resolve(paths["interim_dir"], repo)
        processed = _resolve(paths["processed_dir"], repo)

    reports = _resolve(
        reports_dir if reports_dir is not None else paths["reports_dir"], repo
    )
    notes = _resolve(paths["data_notes"], repo)
    if reports_dir is not None:
        # Keep data notes with the rest of the Colab outputs when reports are redirected.
        notes = reports.parent / "docs" / "data_notes.md"

    sample = (
        int(sample_rows)
        if sample_rows is not None
        else _as_optional_int(raw.get("sample_rows"))
    )
    keywords_broad = [str(item) for item in raw["keywords_broad"]]
    keywords_narrow = [str(item) for item in raw["keywords_narrow"]]
    use_narrow = bool(raw.get("use_narrow_lexicon", False))

    return Model1Config(
        raw=raw,
        repo_root=repo,
        dataset_name=str(raw["dataset"]["name"]),
        dataset_revision=str(raw["dataset"]["revision"]),
        dataset_files=[str(item) for item in raw["dataset"]["files"]],
        columns_keep=[str(item) for item in raw["dataset"]["columns_keep"]],
        finbert=_model_spec("finbert", raw["models"]["finbert"]),
        inflabert=_model_spec("inflabert", raw["models"]["inflabert"]),
        cutoff_day=int(raw["cutoff_day"]),
        min_articles_per_month=int(raw["min_articles_per_month"]),
        consecutive_months=int(raw["consecutive_months"]),
        batch_size=int(raw["batch_size"]),
        max_length=int(raw["max_length"]),
        chunksize=int(raw["chunksize"]),
        sample_rows=sample,
        use_narrow_lexicon=use_narrow,
        prefer_lsa_summary=bool(raw.get("prefer_lsa_summary", False)),
        sanity_sample_size=int(raw.get("sanity_sample_size", 50)),
        sanity_sample_seed=int(raw.get("sanity_sample_seed", 42)),
        keywords=keywords_narrow if use_narrow else keywords_broad,
        keywords_broad=keywords_broad,
        keywords_narrow=keywords_narrow,
        data_root=root,
        interim_dir=interim,
        processed_dir=processed,
        reports_dir=reports,
        data_notes_path=notes,
        cpi_url=str(raw["cpi"]["url"]),
        cpi_date_col=str(raw["cpi"]["date_col"]),
        cpi_value_col=str(raw["cpi"]["value_col"]),
    )
