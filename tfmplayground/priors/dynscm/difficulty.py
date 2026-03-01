"""Research-only learnability and difficulty utilities for DynSCM samples."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

DifficultyBin = Literal["easy", "medium", "hard"]


@dataclass(frozen=True, slots=True)
class DynSCMLearnabilityMetrics:
    train_target_std: float
    test_target_std: float
    max_abs_target_value: float
    informative_feature_count: int
    informative_feature_std_floor: float
    probe_r2_train: float | None
    probe_r2_holdout: float | None
    missing_fraction: float
    block_missing_fraction: float
    difficulty_bin: DifficultyBin

    @property
    def probe_r2(self) -> float | None:
        return self.probe_r2_holdout


def _as_row_matrix(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {arr.shape}.")
    return arr


def _as_vector(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr[0]
    if arr.ndim == 1:
        return arr
    raise ValueError(f"Expected 1D target array, got shape {arr.shape}.")


def safe_std(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1))


def count_informative_features(
    x: np.ndarray,
    *,
    n_train: int,
    std_floor: float,
) -> int:
    train_x = _as_row_matrix(x)[:n_train]
    if train_x.size == 0:
        return 0
    stds = np.std(train_x, axis=0, ddof=1 if train_x.shape[0] > 1 else 0)
    return int(np.sum(stds >= float(std_floor)))


def missing_fraction(observed_mask: np.ndarray) -> float:
    observed = np.asarray(observed_mask, dtype=bool)
    if observed.size == 0:
        return 0.0
    return float(1.0 - observed.mean())


def max_block_missing_fraction(observed_mask: np.ndarray) -> float:
    observed = np.asarray(observed_mask, dtype=bool)
    if observed.ndim == 2:
        observed = observed[None, :, :]
    if observed.ndim != 3:
        raise ValueError(
            f"Expected observed_mask with shape (B, T, p), got {observed.shape}."
        )
    _, num_steps, num_vars = observed.shape
    if num_steps == 0 or num_vars == 0:
        return 0.0

    longest = 0
    for batch_idx in range(observed.shape[0]):
        for var_idx in range(num_vars):
            current = 0
            for is_observed in observed[batch_idx, :, var_idx].tolist():
                if is_observed:
                    current = 0
                    continue
                current += 1
                if current > longest:
                    longest = current
    return float(longest) / float(num_steps)


def _ridge_design_matrices(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_train: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_matrix = _as_row_matrix(x)
    y_vector = _as_vector(y)
    if x_matrix.shape[0] != y_vector.shape[0]:
        raise ValueError("x and y must agree on row dimension.")
    if n_train <= 1 or n_train > x_matrix.shape[0]:
        raise ValueError("n_train must satisfy 1 < n_train <= num_rows.")

    x_train = x_matrix[:n_train]
    y_train = y_vector[:n_train]
    mean = np.mean(x_train, axis=0, keepdims=True)
    std = np.std(
        x_train,
        axis=0,
        ddof=1 if x_train.shape[0] > 1 else 0,
        keepdims=True,
    )
    safe_std_arr = np.where(std >= 1e-8, std, 1.0)
    x_train_std = (x_train - mean) / safe_std_arr
    return x_train_std, y_train, mean, safe_std_arr


def _ridge_predict(
    x_train_std: np.ndarray,
    y_train: np.ndarray,
    x_eval_std: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    if x_eval_std.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)

    design_train = np.concatenate(
        [np.ones((x_train_std.shape[0], 1), dtype=np.float64), x_train_std],
        axis=1,
    )
    design_eval = np.concatenate(
        [np.ones((x_eval_std.shape[0], 1), dtype=np.float64), x_eval_std],
        axis=1,
    )
    ridge = np.eye(design_train.shape[1], dtype=np.float64)
    ridge[0, 0] = 0.0
    gram = design_train.T @ design_train + (float(alpha) * ridge)
    rhs = design_train.T @ y_train
    coeffs = np.linalg.solve(gram, rhs)
    return design_eval @ coeffs


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    sse = float(np.sum((y_true - y_pred) ** 2))
    centered = y_true - float(np.mean(y_true))
    sst = float(np.sum(centered**2))
    if sst <= 1e-12:
        return 0.0
    return float(max(-1.0, min(1.0, 1.0 - (sse / sst))))


def ridge_probe_r2_train(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_train: int,
    alpha: float = 1e-3,
) -> float:
    x_train_std, y_train, _, _ = _ridge_design_matrices(x, y, n_train=n_train)
    predictions = _ridge_predict(
        x_train_std,
        y_train,
        x_train_std,
        alpha=alpha,
    )
    return _r2_score(y_train, predictions)


def ridge_probe_r2_holdout(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_train: int,
    alpha: float = 1e-3,
) -> float:
    x_matrix = _as_row_matrix(x)
    y_vector = _as_vector(y)
    if n_train >= x_matrix.shape[0]:
        return 0.0
    x_train_std, y_train, mean, safe_std_arr = _ridge_design_matrices(
        x_matrix,
        y_vector,
        n_train=n_train,
    )
    x_test = x_matrix[n_train:]
    y_test = y_vector[n_train:]
    if x_test.shape[0] == 0:
        return 0.0
    x_test_std = (x_test - mean) / safe_std_arr
    predictions = _ridge_predict(
        x_train_std,
        y_train,
        x_test_std,
        alpha=alpha,
    )
    return _r2_score(y_test, predictions)


def ridge_holdout_predictions(
    x: np.ndarray,
    y: np.ndarray,
    *,
    n_train: int,
    alpha: float = 1e-3,
) -> np.ndarray:
    x_matrix = _as_row_matrix(x)
    y_vector = _as_vector(y)
    if x_matrix.shape[0] != y_vector.shape[0]:
        raise ValueError("x and y must agree on row dimension.")
    if n_train <= 1 or n_train >= x_matrix.shape[0]:
        return np.zeros((max(x_matrix.shape[0] - n_train, 0),), dtype=np.float64)
    x_train_std, y_train, mean, safe_std_arr = _ridge_design_matrices(
        x_matrix,
        y_vector,
        n_train=n_train,
    )
    x_test = x_matrix[n_train:]
    if x_test.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    x_test_std = (x_test - mean) / safe_std_arr
    return _ridge_predict(
        x_train_std,
        y_train,
        x_test_std,
        alpha=alpha,
    )


def classify_difficulty(
    *,
    probe_r2: float | None,
    train_target_std: float,
    has_long_history: bool,
    has_regimes: bool,
    has_drift: bool,
    has_missingness: bool,
    has_heavy_tails: bool,
) -> DifficultyBin:
    hard_factor_count = sum(
        (
            bool(has_long_history),
            bool(has_regimes),
            bool(has_drift),
            bool(has_missingness),
            bool(has_heavy_tails),
        )
    )
    if train_target_std < 1e-3:
        return "hard"
    if probe_r2 is not None:
        if probe_r2 < 0.02:
            return "hard"
        if probe_r2 >= 0.10 and hard_factor_count <= 1:
            return "easy"
        if probe_r2 >= 0.05 and hard_factor_count <= 2:
            return "medium"
        return "hard"
    if hard_factor_count <= 1:
        return "easy"
    if hard_factor_count <= 2:
        return "medium"
    return "hard"


def measure_sample_learnability(
    *,
    x: np.ndarray,
    y: np.ndarray,
    n_train: int,
    observed_mask: np.ndarray,
    informative_feature_std_floor: float,
    compute_probe_r2: bool,
    has_long_history: bool,
    has_regimes: bool,
    has_drift: bool,
    has_missingness: bool,
    has_heavy_tails: bool,
) -> DynSCMLearnabilityMetrics:
    y_vector = _as_vector(y)
    train_std = safe_std(y_vector[:n_train])
    test_std = safe_std(y_vector[n_train:])
    probe_r2_train = (
        ridge_probe_r2_train(x, y_vector, n_train=n_train) if compute_probe_r2 else None
    )
    probe_r2_holdout = (
        ridge_probe_r2_holdout(x, y_vector, n_train=n_train)
        if compute_probe_r2
        else None
    )
    informative_count = count_informative_features(
        x,
        n_train=n_train,
        std_floor=informative_feature_std_floor,
    )
    missing_frac = missing_fraction(observed_mask)
    block_missing_frac = max_block_missing_fraction(observed_mask)
    difficulty_bin = classify_difficulty(
        probe_r2=probe_r2_holdout,
        train_target_std=train_std,
        has_long_history=has_long_history,
        has_regimes=has_regimes,
        has_drift=has_drift,
        has_missingness=has_missingness,
        has_heavy_tails=has_heavy_tails,
    )
    return DynSCMLearnabilityMetrics(
        train_target_std=train_std,
        test_target_std=test_std,
        max_abs_target_value=float(np.max(np.abs(y_vector))) if y_vector.size else 0.0,
        informative_feature_count=informative_count,
        informative_feature_std_floor=float(informative_feature_std_floor),
        probe_r2_train=probe_r2_train,
        probe_r2_holdout=probe_r2_holdout,
        missing_fraction=missing_frac,
        block_missing_fraction=block_missing_frac,
        difficulty_bin=difficulty_bin,
    )
