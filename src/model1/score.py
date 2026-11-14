"""Step 3: batched FinBERT + InflaBERT scoring with resumable chunk writes."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.model1.config import Model1Config, ModelSpec
from src.model1.labels import (
    FINBERT_PROB_COLUMNS,
    INFLA_PROB_COLUMNS,
    RoleMap,
    decode_logits,
    decoded_to_frame,
    resolve_role_map,
)
from src.model1.load import write_parquet_atomic

LOGGER = logging.getLogger(__name__)

SCORE_COLUMNS = [
    "finbert_p_pos",
    "finbert_p_neg",
    "finbert_p_neu",
    "finbert_label",
    "finbert_role",
    "finbert_score",
    "infla_p_up",
    "infla_p_down",
    "infla_p_neu",
    "infla_label",
    "infla_role",
    "infla_score",
]
CATEGORY_COLUMNS = [
    "Stock_symbol",
    "Publisher",
    "source_file",
    "finbert_label",
    "finbert_role",
    "infla_label",
    "infla_role",
]
UNION_COLUMNS = ["Date", "Article_title", "Stock_symbol", "Publisher", "source_file", *SCORE_COLUMNS]


def active_score_columns(cfg: Model1Config) -> list[str]:
    """SCORE_COLUMNS for the enabled models (FinBERT columns drop out when disabled)."""
    if cfg.finbert_enabled:
        return list(SCORE_COLUMNS)
    return [column for column in SCORE_COLUMNS if not column.startswith("finbert_")]


@dataclass
class LoadedScorer:
    spec: ModelSpec
    tokenizer: object
    model: object
    role_map: RoleMap
    device: torch.device
    prefix: str
    prob_columns: dict[str, str]


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        LOGGER.info("Using CUDA device: %s", name)
        return torch.device("cuda")
    LOGGER.warning("No GPU detected; scoring on CPU. On Colab: Runtime > Change runtime type > T4 GPU.")
    return torch.device("cpu")


def load_scorer(
    spec: ModelSpec,
    *,
    device: torch.device | None = None,
    prefix: str,
    prob_columns: dict[str, str],
) -> LoadedScorer:
    device = device or pick_device()
    LOGGER.info("Loading %s (%s @ %s) on %s", spec.key, spec.name, spec.revision, device)
    tokenizer = AutoTokenizer.from_pretrained(spec.name, revision=spec.revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.name, revision=spec.revision
    )
    model.to(device)
    model.eval()
    id2label = dict(model.config.id2label)
    role_map = resolve_role_map(
        id2label,
        spec.role_aliases,
        positive_role=spec.score_positive_role,
        negative_role=spec.score_negative_role,
        model_name=f"{spec.key} ({spec.name})",
    )
    return LoadedScorer(
        spec=spec,
        tokenizer=tokenizer,
        model=model,
        role_map=role_map,
        device=device,
        prefix=prefix,
        prob_columns=prob_columns,
    )


def load_both_scorers(cfg: Model1Config, device: torch.device | None = None) -> dict[str, LoadedScorer]:
    """Load the enabled scorers. FinBERT is skipped when models.finbert.enabled is false."""
    device = device or pick_device()
    scorers: dict[str, LoadedScorer] = {}
    if cfg.finbert_enabled:
        scorers["finbert"] = load_scorer(
            cfg.finbert,
            device=device,
            prefix="finbert",
            prob_columns=FINBERT_PROB_COLUMNS,
        )
    else:
        LOGGER.info("FinBERT disabled (models.finbert.enabled=false); InflaBERT only.")
    scorers["inflabert"] = load_scorer(
        cfg.inflabert,
        device=device,
        prefix="infla",
        prob_columns=INFLA_PROB_COLUMNS,
    )
    return scorers


def texts_to_score(frame: pd.DataFrame, *, prefer_lsa_summary: bool) -> list[str]:
    titles = frame["Article_title"].astype("string").fillna("").tolist()
    if not prefer_lsa_summary or "Lsa_summary" not in frame.columns:
        return titles
    summaries = frame["Lsa_summary"].astype("string").fillna("")
    out: list[str] = []
    for title, summary in zip(titles, summaries, strict=False):
        text = summary.strip()
        out.append(text if text else title)
    return out


def score_texts(
    texts: list[str],
    scorer: LoadedScorer,
    *,
    batch_size: int,
    max_length: int,
    use_fp16: bool = False,
) -> pd.DataFrame:
    if not texts:
        columns = [
            f"{scorer.prefix}_label",
            f"{scorer.prefix}_role",
            f"{scorer.prefix}_score",
            *scorer.prob_columns.values(),
        ]
        return pd.DataFrame(columns=columns)

    # Batch texts of similar length so padding (and GPU time) stays small; the
    # original order is restored at the end.
    order = sorted(range(len(texts)), key=lambda idx: len(texts[idx]))
    ordered = [texts[idx] or " " for idx in order]
    precision = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if scorer.device.type == "cuda" and use_fp16
        else contextlib.nullcontext()
    )

    pieces: list[pd.DataFrame] = []
    model = scorer.model
    tokenizer = scorer.tokenizer
    with torch.no_grad(), precision:
        for start in range(0, len(ordered), batch_size):
            encoded = tokenizer(
                ordered[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(scorer.device) for key, value in encoded.items()}
            logits = model(**encoded).logits.detach().float().cpu().numpy()
            decoded = decode_logits(logits, scorer.role_map)
            pieces.append(
                decoded_to_frame(
                    decoded, prefix=scorer.prefix, role_prob_names=scorer.prob_columns
                )
            )
    scored = pd.concat(pieces, ignore_index=True)
    scored.index = order
    return scored.sort_index().reset_index(drop=True)


def score_frame(
    frame: pd.DataFrame,
    scorers: dict[str, LoadedScorer],
    cfg: Model1Config,
) -> pd.DataFrame:
    texts = texts_to_score(frame, prefer_lsa_summary=cfg.prefer_lsa_summary)
    scored = frame.reset_index(drop=True).copy()
    for key, scorer in scorers.items():
        if key == "finbert" and not cfg.finbert_enabled:
            continue
        cols = score_texts(
            texts,
            scorer,
            batch_size=cfg.batch_size,
            max_length=cfg.max_length,
            use_fp16=cfg.use_fp16,
        )
        scored = pd.concat([scored, cols], axis=1)
    return scored


def _filtered_chunk_paths(cfg: Model1Config) -> list[Path]:
    return sorted(cfg.filtered_dir.glob("*/chunk_*.parquet"))


def _scored_path_for(cfg: Model1Config, filtered_path: Path) -> Path:
    relative = filtered_path.relative_to(cfg.filtered_dir)
    return cfg.scored_dir / relative


def score_filtered_chunks(cfg: Model1Config) -> pd.DataFrame:
    """Score each filtered chunk unless the matching scored parquet already exists."""
    paths = _filtered_chunk_paths(cfg)
    if not paths:
        raise FileNotFoundError(
            f"No filtered chunks under {cfg.filtered_dir}. Run stream_and_filter first."
        )

    pending = [path for path in paths if not _scored_path_for(cfg, path).exists()]
    LOGGER.info("Scoring %s/%s filtered chunks.", len(pending), len(paths))
    scorers = None
    if pending:
        scorers = load_both_scorers(cfg)

    for path in pending:
        dest = _scored_path_for(cfg, path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        chunk = pd.read_parquet(path)
        if chunk.empty:
            scored = chunk.copy()
            for column in active_score_columns(cfg):
                if column.endswith("_label") or column.endswith("_role"):
                    scored[column] = pd.Series(dtype="string")
                else:
                    scored[column] = pd.Series(dtype="float64")
        else:
            scored = score_frame(chunk, scorers, cfg)
        write_parquet_atomic(scored, dest)
        LOGGER.info("Wrote %s (%s rows).", dest, len(scored))

    # Lsa_summary is dropped here: the full run has millions of rows and the
    # summaries alone would not fit in Colab's RAM. Later steps never use it.
    frames = []
    for path in paths:
        chunk = pd.read_parquet(_scored_path_for(cfg, path))
        chunk = chunk.loc[:, [col for col in UNION_COLUMNS if col in chunk.columns]]
        for column in CATEGORY_COLUMNS:
            if column in chunk.columns:
                chunk[column] = chunk[column].astype("string")
        frames.append(chunk)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    for column in CATEGORY_COLUMNS:
        if column in combined.columns:
            combined[column] = combined[column].astype("category")
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cfg.news_scored_path, index=False)
    LOGGER.info("Wrote union scored file %s (%s rows).", cfg.news_scored_path, len(combined))
    return combined
