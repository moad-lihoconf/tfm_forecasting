"""Metrics for forecasting regression and proxy classification benchmarks."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

__all__ = [
    "bootstrap_ci",
    "compute_proxy_metrics",
    "compute_regression_metrics",
    "mase",
    "relative_improvement",
    "rmse",
    "smape",
]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean(err**2)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    denom = np.abs(y_true_arr) + np.abs(y_pred_arr)
    safe = np.where(denom > 0.0, denom, 1.0)
    value = 2.0 * np.abs(y_true_arr - y_pred_arr) / safe
    return float(np.mean(value))


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    insample: np.ndarray,
    seasonality: int,
) -> float:
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)
    insample_arr = np.asarray(insample, dtype=np.float64)

    m = max(int(seasonality), 1)
    if insample_arr.size <= m:
        scale = np.mean(np.abs(np.diff(insample_arr))) if insample_arr.size > 1 else 1.0
    else:
        scale = np.mean(np.abs(insample_arr[m:] - insample_arr[:-m]))
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0

    return float(np.mean(np.abs(y_true_arr - y_pred_arr)) / scale)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    insample: np.ndarray,
    seasonality: int,
) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "mase": mase(y_true, y_pred, insample=insample, seasonality=seasonality),
    }


def compute_proxy_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None,
) -> dict[str, float]:
    true_arr = np.asarray(y_true)
    pred_arr = np.asarray(y_pred)

    bal_acc = float(balanced_accuracy_score(true_arr, pred_arr))

    auc = np.nan
    if y_proba is not None:
        proba_arr = np.asarray(y_proba, dtype=np.float64)
        try:
            if proba_arr.ndim == 1:
                auc = float(roc_auc_score(true_arr, proba_arr))
            else:
                auc = float(roc_auc_score(true_arr, proba_arr, multi_class="ovr"))
        except ValueError:
            auc = np.nan

    return {
        "balanced_accuracy": bal_acc,
        "macro_auroc": float(auc),
    }


def relative_improvement(
    baseline_error: np.ndarray,
    candidate_error: np.ndarray,
) -> np.ndarray:
    """Compute error reduction: positive means candidate is better."""
    baseline = np.asarray(baseline_error, dtype=np.float64)
    candidate = np.asarray(candidate_error, dtype=np.float64)
    denom = np.where(np.abs(baseline) > 1e-12, baseline, 1.0)
    return (baseline - candidate) / denom


def bootstrap_ci(
    values: np.ndarray,
    *,
    n_bootstrap: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("values must be a 1D array.")
    if arr.size == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    n = arr.size
    idxs = rng.integers(0, n, size=(n_bootstrap, n))
    sample_means = np.mean(arr[idxs], axis=1)

    alpha = 1.0 - confidence_level
    lower = float(np.quantile(sample_means, alpha / 2.0))
    upper = float(np.quantile(sample_means, 1.0 - alpha / 2.0))
    mean = float(np.mean(arr))
    return mean, lower, upper
