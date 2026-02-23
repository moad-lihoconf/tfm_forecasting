"""Proxy classification utilities for forecast-target discretization."""

from __future__ import annotations

import numpy as np

__all__ = ["fit_quantile_binner", "transform_to_classes"]


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
    if y.size < max(num_classes * min_samples_per_class, 2):
        raise ValueError("Not enough train samples to fit quantile binner.")

    q = np.linspace(0.0, 1.0, num_classes + 1)
    edges = np.quantile(y, q)
    edges = np.unique(edges)
    if edges.size < 3:
        raise ValueError("Quantile binning collapsed to <2 classes.")

    # Use open-ended boundaries to avoid clipping losses.
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges.astype(np.float64)


def transform_to_classes(y: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map continuous values to class IDs in [0, C-1]."""
    values = np.asarray(y, dtype=np.float64)
    bins = np.asarray(edges, dtype=np.float64)
    if bins.ndim != 1 or bins.size < 3:
        raise ValueError("edges must be a 1D array with at least 3 values.")

    # np.digitize returns 1..len(edges)-1 for bins. Shift to 0-index.
    labels = np.digitize(values, bins[1:-1], right=False)
    return labels.astype(np.int64)
