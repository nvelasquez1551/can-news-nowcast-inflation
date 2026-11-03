"""Forecast accuracy: RMSE, MAE, and the Diebold-Mariano test (HLN correction)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

LOGGER = logging.getLogger(__name__)

ALTERNATIVES = ("two-sided", "less", "greater")


def _as_array(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def rmse(y_true, y_pred) -> float:
    errors = _as_array(y_true) - _as_array(y_pred)
    return float(np.sqrt(np.mean(errors**2)))


def mae(y_true, y_pred) -> float:
    errors = _as_array(y_true) - _as_array(y_pred)
    return float(np.mean(np.abs(errors)))


@dataclass(frozen=True)
class DMResult:
    statistic: float
    pvalue: float
    n: int
    horizon: int
    alternative: str


def diebold_mariano(
    errors_a,
    errors_b,
    *,
    horizon: int = 1,
    loss: str = "squared",
    alternative: str = "two-sided",
) -> DMResult:
    """Test equal accuracy of forecasts A and B from their errors (y - forecast).

    d_t = L(e_a) - L(e_b). A positive statistic means A has the larger loss, so B is
    more accurate. alternative="greater" tests exactly that (B better than A).
    Uses the Harvey-Leybourne-Newbold small-sample correction and a t(n-1) p-value.
    """
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}, got {alternative!r}.")
    if horizon < 1:
        raise ValueError("horizon must be >= 1.")
    e_a = _as_array(errors_a)
    e_b = _as_array(errors_b)
    if e_a.shape != e_b.shape or e_a.ndim != 1:
        raise ValueError("errors_a and errors_b must be 1-D arrays of the same length.")
    if np.isnan(e_a).any() or np.isnan(e_b).any():
        raise ValueError("Errors contain NaN; align both forecasts on the same months first.")

    if loss == "squared":
        d = e_a**2 - e_b**2
    elif loss == "absolute":
        d = np.abs(e_a) - np.abs(e_b)
    else:
        raise ValueError(f"loss must be 'squared' or 'absolute', got {loss!r}.")

    n = len(d)
    if n <= horizon:
        raise ValueError(f"Need more than {horizon} observations, got {n}.")

    d_bar = d.mean()
    centered = d - d_bar
    # Long-run variance with autocovariances up to lag horizon-1 (MA(h-1) forecast errors).
    long_run_var = np.mean(centered**2)
    for lag in range(1, horizon):
        long_run_var += 2 * np.sum(centered[lag:] * centered[:-lag]) / n

    if long_run_var <= 0:
        LOGGER.warning("Loss differential has zero variance; DM statistic is undefined.")
        return DMResult(np.nan, np.nan, n, horizon, alternative)

    dm = d_bar / np.sqrt(long_run_var / n)
    hln = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    statistic = float(dm * hln)

    dist = stats.t(df=n - 1)
    if alternative == "two-sided":
        pvalue = 2 * dist.sf(abs(statistic))
    elif alternative == "greater":
        pvalue = dist.sf(statistic)
    else:
        pvalue = dist.cdf(statistic)
    return DMResult(statistic, float(pvalue), n, horizon, alternative)


def compare_forecasts(
    frame: pd.DataFrame,
    *,
    target: str,
    forecasts: list[str],
    baseline: str,
    horizon: int = 1,
    alternative: str = "greater",
) -> pd.DataFrame:
    """One row per forecast column: n, RMSE, MAE, and DM against the baseline.

    Rows with NaN in the target or any forecast are dropped first, so every
    forecast is scored on exactly the same months. With the default
    alternative="greater", a small p-value means the forecast beats the baseline.
    """
    columns = [target, baseline, *[col for col in forecasts if col != baseline]]
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise KeyError(f"Columns not in frame: {missing}")

    aligned = frame.loc[:, columns].dropna()
    n_dropped = len(frame) - len(aligned)
    if n_dropped:
        LOGGER.info("Dropped %s rows with NaN so all forecasts share the same months.", n_dropped)
    if aligned.empty:
        raise ValueError("No rows left after aligning target and forecasts.")

    y = aligned[target].to_numpy(dtype=np.float64)
    base_errors = y - aligned[baseline].to_numpy(dtype=np.float64)
    rows = []
    for col in columns[1:]:
        pred = aligned[col].to_numpy(dtype=np.float64)
        row = {
            "forecast": col,
            "n": len(aligned),
            "rmse": rmse(y, pred),
            "mae": mae(y, pred),
            "rmse_vs_baseline": rmse(y, pred) / rmse(y, aligned[baseline]),
            "dm_stat": np.nan,
            "dm_pvalue": np.nan,
        }
        if col != baseline:
            result = diebold_mariano(
                base_errors, y - pred, horizon=horizon, alternative=alternative
            )
            row["dm_stat"] = result.statistic
            row["dm_pvalue"] = result.pvalue
        rows.append(row)
    return pd.DataFrame(rows)
