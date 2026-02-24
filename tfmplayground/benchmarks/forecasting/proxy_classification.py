"""Proxy classification utilities for forecast-target discretization."""

from __future__ import annotations

from typing import Literal

import numpy as np

__all__ = ["choose_num_classes", "fit_quantile_binner", "transform_to_classes"]


def choose_num_classes(
    y_train: np.ndarray,
    *,
    num_classes: int | Literal["auto"],
    min_samples_per_class: int,
    cube_root_scale: float = 1.5,
) -> int:
    """Choose a feasible class count from train-only labels.

    Heuristic:
    - If `num_classes == "auto"`, use round(kappa * n^(1/3)).
    - Clamp by feasibility `<= floor(n / min_samples_per_class)`.
    - Enforce at least 2 classes, else raise.
    """
    if min_samples_per_class < 1:
        raise ValueError("min_samples_per_class must be >= 1.")

    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    n_train = int(y.size)
    if n_train < 2:
        raise ValueError("Need at least 2 train samples to build classes.")

    feasible_max = n_train // int(min_samples_per_class)
    if feasible_max < 2:
        raise ValueError(
            "Not enough train samples for at least 2 classes under "
            "min_samples_per_class."
        )

    if num_classes == "auto":
        # n^(1/3) scaling: quantization bias ~ O(C^-2), class estimation
        # variance ~ O(C / n), balanced by C ~ n^(1/3).
        suggested = int(np.round(cube_root_scale * np.cbrt(float(n_train))))
        requested = max(2, suggested)
    else:
        requested = int(num_classes)
        if requested < 2:
            raise ValueError("num_classes must be >= 2.")

    return int(min(requested, feasible_max))


def fit_quantile_binner(
    y_train: np.ndarray,
    *,
    num_classes: int,
    min_samples_per_class: int,
) -> np.ndarray:
    """Fit quantile bin edges using train-only targets."""
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2.")

    y = np.asarray(y_train, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError("y_train must be a 1D array.")

    # Attempt requested class count first, then back off deterministically if
    # quantiles collapse on tied targets.
    for cls_count in range(int(num_classes), 1, -1):
        if y.size < max(cls_count * min_samples_per_class, 2):
            continue
        q = np.linspace(0.0, 1.0, cls_count + 1)
        edges = np.quantile(y, q)
        edges = np.unique(edges)
        if edges.size < 3:
            continue

        # Use open-ended boundaries to avoid clipping losses.
        edges[0] = -np.inf
        edges[-1] = np.inf
        return edges.astype(np.float64)

    raise ValueError(
        "Quantile binning could not produce >=2 classes for requested settings."
    )


def transform_to_classes(y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map continuous values to class IDs in [0, C-1]."""
    values = np.asarray(y, dtype=np.float64)
    bins = np.asarray(edges, dtype=np.float64)
    if bins.ndim != 1 or bins.size < 3:
        raise ValueError("edges must be a 1D array with at least 3 values.")

    # np.digitize returns 1..len(edges)-1 for bins. Shift to 0-index.
    labels = np.digitize(values, bins[1:-1], right=False)
    return labels.astype(np.int64)
