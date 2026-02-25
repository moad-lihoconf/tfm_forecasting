"""Rolling-origin split protocol for forecasting benchmark evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "RollingOriginIndices",
    "generate_rolling_origin_indices",
    "validate_indices",
]


@dataclass(frozen=True, slots=True)
class RollingOriginIndices:
    """Origin and horizon indices for one forecast table."""

    t_idx: np.ndarray  # shape: (N,)
    h_idx: np.ndarray  # shape: (N,)
    n_train: int
    n_test: int

    @property
    def split_index(self) -> int:
        return self.n_train


def generate_rolling_origin_indices(
    *,
    series_length: int,
    horizons: tuple[int, ...],
    n_train: int,
    n_test: int,
    required_lag: int,
    seed: int,
) -> RollingOriginIndices:
    """Generate chronology-safe rolling-origin rows for one time series."""
    if series_length < 2:
        raise ValueError("series_length must be >= 2.")
    if n_train < 1 or n_test < 1:
        raise ValueError("n_train and n_test must be >= 1.")
    if required_lag < 0:
        raise ValueError("required_lag must be >= 0.")

    horizon_arr = np.asarray(horizons, dtype=np.int64)
    if horizon_arr.ndim != 1 or horizon_arr.size == 0:
        raise ValueError("horizons must be a non-empty 1D tuple.")
    if np.any(horizon_arr <= 0):
        raise ValueError("horizons must contain positive integers.")

    max_horizon = int(np.max(horizon_arr))
    min_origin = required_lag
    max_origin = series_length - 1 - max_horizon
    if max_origin <= min_origin + 1:
        raise ValueError(
            "Insufficient timeline budget for strict chronology"
            " and requested horizons: "
            f"need max_origin > min_origin + 1, "
            f"got {max_origin=} and {min_origin=}."
        )

    rng = np.random.default_rng(seed)

    split_origin = int(rng.integers(min_origin + 1, max_origin + 1))
    train_low = min_origin
    train_high = split_origin - 1
    test_low = split_origin
    test_high = max_origin
    if train_high < train_low or test_high < test_low:
        raise ValueError(
            "Could not construct chronology-safe train/test origin ranges."
        )

    t_train = _sample_ints(rng, low=train_low, high=train_high, size=n_train)
    t_test = _sample_ints(rng, low=test_low, high=test_high, size=n_test)

    h_train = rng.choice(horizon_arr, size=n_train, replace=True)
    h_test = rng.choice(horizon_arr, size=n_test, replace=True)

    t_idx = np.concatenate([t_train, t_test]).astype(np.int64)
    h_idx = np.concatenate([h_train, h_test]).astype(np.int64)

    validate_indices(
        t_idx=t_idx,
        h_idx=h_idx,
        n_train=n_train,
        series_length=series_length,
        required_lag=required_lag,
    )
    return RollingOriginIndices(
        t_idx=t_idx, h_idx=h_idx, n_train=n_train, n_test=n_test
    )


def validate_indices(
    *,
    t_idx: np.ndarray,
    h_idx: np.ndarray,
    n_train: int,
    series_length: int,
    required_lag: int,
) -> None:
    """Validate chronology, lag safety, and label index safety."""
    if t_idx.shape != h_idx.shape:
        raise ValueError("t_idx and h_idx must have the same shape.")
    if t_idx.ndim != 1:
        raise ValueError("t_idx and h_idx must be 1D arrays.")
    if t_idx.size <= n_train:
        raise ValueError("Need at least one test row: len(t_idx) must be > n_train.")
    if np.any(t_idx < required_lag):
        raise ValueError("Found origin index below required lag floor.")
    if np.any(h_idx <= 0):
        raise ValueError("All horizons must be positive.")
    if np.any(t_idx + h_idx >= series_length):
        raise ValueError("Found label index outside timeline: t_idx + h_idx >= T.")

    train_max = int(np.max(t_idx[:n_train]))
    test_min = int(np.min(t_idx[n_train:]))
    if train_max >= test_min:
        raise ValueError(
            "Strict chronology violated: max(train origins) >= min(test origins)."
        )


def _sample_ints(
    rng: np.random.Generator,
    *,
    low: int,
    high: int,
    size: int,
) -> np.ndarray:
    """Sample inclusive integer range with replacement."""
    if size < 1:
        raise ValueError("size must be >= 1.")
    if high < low:
        raise ValueError("high must be >= low.")
    return rng.integers(low, high + 1, size=size, dtype=np.int64)
