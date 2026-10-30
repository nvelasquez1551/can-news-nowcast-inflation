"""Map model.config.id2label onto score roles and decode logits.

Label *order* is never assumed. Tests pass a shuffled id2label to prove that.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

PROB_ATOL = 1e-6


@dataclass(frozen=True)
class RoleMap:
    id2label: dict[int, str]
    role_to_index: dict[str, int]
    index_to_role: dict[int, str]
    positive_role: str
    negative_role: str

    def label_at(self, class_index: int) -> str:
        return self.id2label[class_index]


def normalize_id2label(id2label: dict) -> dict[int, str]:
    return {int(key): str(value) for key, value in id2label.items()}


def _alias_lookup(role_aliases: dict[str, list[str]]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for role, aliases in role_aliases.items():
        for alias in aliases:
            key = str(alias).strip().lower()
            if key in lookup and lookup[key] != role:
                raise ValueError(
                    f"Alias {alias!r} maps to both {lookup[key]!r} and {role!r}."
                )
            lookup[key] = role
    return lookup


def resolve_role_map(
    id2label: dict,
    role_aliases: dict[str, list[str]],
    *,
    positive_role: str,
    negative_role: str,
    model_name: str = "model",
) -> RoleMap:
    """Bijection: each class label matches exactly one role, each role one class."""
    normalized = normalize_id2label(id2label)
    lookup = _alias_lookup(role_aliases)
    role_to_index: dict[str, int] = {}
    index_to_role: dict[int, str] = {}

    for class_index, label in sorted(normalized.items()):
        role = lookup.get(label.strip().lower())
        if role is None:
            raise ValueError(
                f"{model_name}: label {label!r} (id={class_index}) is not in role_aliases "
                f"{ {k: v for k, v in role_aliases.items()} }."
            )
        if role in role_to_index:
            other = role_to_index[role]
            raise ValueError(
                f"{model_name}: role {role!r} matched both id {other} "
                f"({normalized[other]!r}) and id {class_index} ({label!r})."
            )
        role_to_index[role] = class_index
        index_to_role[class_index] = role

    expected_roles = set(role_aliases)
    missing = expected_roles - set(role_to_index)
    extra = set(role_to_index) - expected_roles
    if missing or extra:
        raise ValueError(
            f"{model_name}: role bijection failed; missing={sorted(missing)} extra={sorted(extra)}."
        )
    if positive_role not in role_to_index or negative_role not in role_to_index:
        raise ValueError(
            f"{model_name}: score roles {positive_role!r}/{negative_role!r} "
            f"not in {sorted(role_to_index)}."
        )

    mapping_txt = ", ".join(
        f"{role}->{idx} ({normalized[idx]})"
        for role, idx in sorted(role_to_index.items(), key=lambda item: item[1])
    )
    LOGGER.info("%s id2label: %s", model_name, normalized)
    LOGGER.info("%s role map: %s", model_name, mapping_txt)

    return RoleMap(
        id2label=normalized,
        role_to_index=role_to_index,
        index_to_role=index_to_role,
        positive_role=positive_role,
        negative_role=negative_role,
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def assert_valid_probs(probs: np.ndarray, *, atol: float = PROB_ATOL) -> None:
    row_sums = np.asarray(probs, dtype=np.float64).sum(axis=-1)
    if not np.allclose(row_sums, 1.0, atol=atol):
        raise ValueError(f"Probabilities must sum to 1; got sums {row_sums}.")
    if np.any(probs < -atol) or np.any(probs > 1.0 + atol):
        raise ValueError("Probabilities must lie in [0, 1].")


def assert_valid_scores(scores: np.ndarray) -> None:
    values = np.asarray(scores, dtype=np.float64)
    if np.any(values < -1.0 - 1e-8) or np.any(values > 1.0 + 1e-8):
        raise ValueError(f"Scores must lie in [-1, 1]; range=({values.min()}, {values.max()}).")


def decode_logits(logits: np.ndarray, role_map: RoleMap) -> dict[str, np.ndarray]:
    """Softmax logits and return per-role probabilities, raw labels, roles, and score."""
    probs = softmax(logits)
    assert_valid_probs(probs)
    pred_idx = probs.argmax(axis=-1)
    n_classes = probs.shape[1]
    if n_classes != len(role_map.id2label):
        raise ValueError(
            f"Logits have {n_classes} classes but id2label has {len(role_map.id2label)}."
        )

    by_role = {
        role: probs[:, idx] for role, idx in role_map.role_to_index.items()
    }
    pos = by_role[role_map.positive_role]
    neg = by_role[role_map.negative_role]
    scores = pos - neg
    assert_valid_scores(scores)

    labels = np.array([role_map.label_at(int(idx)) for idx in pred_idx])
    roles = np.array([role_map.index_to_role[int(idx)] for idx in pred_idx])
    return {
        "probs": probs,
        "by_role": by_role,
        "label": labels,
        "role": roles,
        "score": scores,
        "pred_index": pred_idx,
    }


def decoded_to_frame(
    decoded: dict[str, np.ndarray],
    *,
    prefix: str,
    role_prob_names: dict[str, str],
) -> pd.DataFrame:
    """Build the spec columns for one scorer, e.g. finbert_p_pos / infla_p_up."""
    data: dict[str, np.ndarray] = {
        f"{prefix}_label": decoded["label"],
        f"{prefix}_role": decoded["role"],
        f"{prefix}_score": decoded["score"],
    }
    for role, column in role_prob_names.items():
        data[column] = decoded["by_role"][role]
    return pd.DataFrame(data)


FINBERT_PROB_COLUMNS = {
    "pos": "finbert_p_pos",
    "neg": "finbert_p_neg",
    "neu": "finbert_p_neu",
}

INFLA_PROB_COLUMNS = {
    "up": "infla_p_up",
    "down": "infla_p_down",
    "neu": "infla_p_neu",
}
