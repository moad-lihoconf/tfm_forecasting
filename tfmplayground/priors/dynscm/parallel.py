"""Shared single-sample generation and worker utilities for DynSCM batches."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

try:
    from threadpoolctl import threadpool_limits
except ModuleNotFoundError:  # pragma: no cover - optional dependency.
    threadpool_limits = None

from .config import DynSCMConfig
from .features import build_forecasting_table, sample_origins_and_horizons
from .graph import sample_regime_graphs
from .mechanisms import sample_regime_mechanisms
from .missingness import sample_observation_mask
from .simulate import simulate_dynscm_series

_SEED_MAX = np.iinfo(np.int64).max
_THREADPOOL_LIMITS = None
_WORKER_CFG: DynSCMConfig | None = None


@dataclass(frozen=True, slots=True)
class DynSCMWorkerTask:
    sample_seed: int
    n_train: int
    n_test: int
    row_budget: int
    num_features: int


def configure_worker_runtime(*, blas_threads: int) -> None:
    """Cap BLAS threads in worker processes to avoid oversubscription."""
    threads = max(1, int(blas_threads))
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
    global _THREADPOOL_LIMITS
    if threadpool_limits is not None:
        _THREADPOOL_LIMITS = threadpool_limits(limits=threads, user_api="blas")


def init_dynscm_worker(
    cfg_payload: Mapping[str, object],
    blas_threads: int,
) -> None:
    """Initializer for process workers."""
    configure_worker_runtime(blas_threads=blas_threads)
    global _WORKER_CFG
    _WORKER_CFG = DynSCMConfig.from_dict(dict(cfg_payload))


def generate_dynscm_worker_sample(
    task: DynSCMWorkerTask,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one sample using worker-local config and task parameters."""
    if _WORKER_CFG is None:
        raise RuntimeError("DynSCM worker was not initialized with config state.")
    return build_single_dynscm_sample(
        _WORKER_CFG,
        sample_seed=task.sample_seed,
        n_train=task.n_train,
        n_test=task.n_test,
        row_budget=task.row_budget,
        num_features=task.num_features,
    )


def draw_seed_bundles(
    rng: np.random.Generator,
    *,
    batch_size: int,
    bundle_width: int,
) -> np.ndarray:
    """Draw a deterministic matrix of int64 seeds."""
    return rng.integers(
        0,
        _SEED_MAX,
        size=(batch_size, bundle_width),
        dtype=np.int64,
    )


def compute_feasible_row_pairs(
    *,
    cfg: DynSCMConfig,
    num_datapoints_max: int,
) -> np.ndarray:
    """Enumerate feasible (n_train, n_test) pairs for a row budget."""
    train_min = max(1, cfg.train_rows_min)
    train_max = max(train_min, cfg.train_rows_max)
    test_min = max(1, cfg.test_rows_min)
    test_max = max(test_min, cfg.test_rows_max)

    feasible_pairs = [
        (n_train, n_test)
        for n_train in range(train_min, train_max + 1)
        for n_test in range(test_min, test_max + 1)
        if n_train + n_test <= num_datapoints_max
    ]
    if not feasible_pairs:
        raise ValueError(
            "No feasible (n_train, n_test) pair satisfies config bounds and "
            f"num_datapoints_max={num_datapoints_max}."
        )

    return np.asarray(feasible_pairs, dtype=np.int64)


def sample_dataset_dimensions(
    cfg: DynSCMConfig,
    rng: np.random.Generator,
) -> tuple[int, int, int]:
    """Sample (num_vars, num_steps, y_idx) for one dataset."""
    num_vars = int(
        rng.integers(
            cfg.num_variables_min,
            cfg.num_variables_max + 1,
        )
    )

    series_len_min = max(cfg.series_length_min, required_min_series_length(cfg))
    series_len_max = cfg.series_length_max
    if series_len_min > series_len_max:
        raise ValueError(
            "series_length_max is too small for feature/horizon safety: "
            f"need at least {series_len_min}, got {series_len_max}."
        )
    num_steps = int(rng.integers(series_len_min, series_len_max + 1))

    y_idx = int(rng.integers(0, num_vars))
    return num_vars, num_steps, y_idx


def required_min_series_length(cfg: DynSCMConfig) -> int:
    required_lag = max(cfg.max_feature_lag, max(cfg.explicit_lags))
    max_horizon = int(max(cfg.forecast_horizons))
    return required_lag + max_horizon + 2


def build_single_dynscm_sample(
    cfg: DynSCMConfig,
    *,
    sample_seed: int,
    n_train: int,
    n_test: int,
    row_budget: int,
    num_features: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate exactly one padded DynSCM `(x_i, y_i)` sample."""
    sample_rng = cfg.make_rng(sample_seed)
    num_vars, num_steps, y_idx = sample_dataset_dimensions(cfg, sample_rng)
    per_sample_seeds = sample_rng.integers(
        0,
        _SEED_MAX,
        size=(6,),
        dtype=np.int64,
    )

    graph_sample = sample_regime_graphs(
        cfg,
        num_vars=num_vars,
        seed=int(per_sample_seeds[0]),
    )
    mechanism_sample = sample_regime_mechanisms(
        cfg,
        graph_sample,
        seed=int(per_sample_seeds[1]),
    )
    simulation_sample = simulate_dynscm_series(
        cfg,
        graph_sample,
        mechanism_sample,
        num_steps=num_steps,
        seed=int(per_sample_seeds[2]),
    )

    series = simulation_sample.series[None, :, :].astype(np.float64, copy=False)
    y_index = np.array([y_idx], dtype=np.int64)

    t_idx, h_idx = sample_origins_and_horizons(
        cfg,
        batch_size=1,
        num_steps=num_steps,
        n_train=n_train,
        n_test=n_test,
        seed=int(per_sample_seeds[3]),
    )
    obs_mask = sample_observation_mask(
        cfg,
        series,
        seed=int(per_sample_seeds[4]),
        label_times=t_idx + h_idx,
        label_var_indices=y_index,
    )

    x_raw, y_raw, metadata = build_forecasting_table(
        cfg,
        series,
        y_index,
        n_train=n_train,
        n_test=n_test,
        t_idx=t_idx,
        h_idx=h_idx,
        obs_mask=obs_mask,
        seed=int(per_sample_seeds[5]),
    )

    x_prioritized = prioritize_feature_blocks(
        x_raw,
        feature_slices=metadata["feature_slices"],
    )
    x_budgeted = fit_feature_budget(x_prioritized, num_features=num_features)
    x_padded = pad_rows_3d(x_budgeted, row_budget=row_budget)
    y_padded = pad_rows_2d(y_raw, row_budget=row_budget)

    return (
        x_padded[0].astype(np.float32, copy=False),
        y_padded[0].astype(np.float32, copy=False),
    )


def slice_or_empty(
    x: np.ndarray,
    feature_slices: Mapping[str, tuple[int, int]],
    key: str,
) -> np.ndarray:
    if key not in feature_slices:
        return np.zeros((x.shape[0], x.shape[1], 0), dtype=x.dtype)
    start, end = feature_slices[key]
    start_idx = int(start)
    end_idx = int(end)
    if end_idx <= start_idx:
        return np.zeros((x.shape[0], x.shape[1], 0), dtype=x.dtype)
    return x[:, :, start_idx:end_idx]


def prioritize_feature_blocks(
    x: np.ndarray,
    *,
    feature_slices: Mapping[str, tuple[int, int]],
) -> np.ndarray:
    """Return features ordered by deterministic priority for truncation safety."""
    blocks = [
        slice_or_empty(x, feature_slices, "deterministic"),
        slice_or_empty(x, feature_slices, "lags_value"),
        slice_or_empty(x, feature_slices, "kern_value"),
        slice_or_empty(x, feature_slices, "data_mask"),
    ]
    non_empty = [block for block in blocks if block.shape[2] > 0]
    if not non_empty:
        return np.zeros((x.shape[0], x.shape[1], 0), dtype=x.dtype)
    return np.concatenate(non_empty, axis=2)


def fit_feature_budget(x: np.ndarray, *, num_features: int) -> np.ndarray:
    if x.shape[2] >= num_features:
        return x[:, :, :num_features]
    pad_width = num_features - x.shape[2]
    return np.pad(x, ((0, 0), (0, 0), (0, pad_width)), mode="constant")


def pad_rows_3d(x: np.ndarray, *, row_budget: int) -> np.ndarray:
    if x.shape[1] > row_budget:
        raise RuntimeError(f"Row count N={x.shape[1]} exceeds row_budget={row_budget}.")
    if x.shape[1] == row_budget:
        return x
    pad_rows = row_budget - x.shape[1]
    return np.pad(x, ((0, 0), (0, pad_rows), (0, 0)), mode="constant")


def pad_rows_2d(y: np.ndarray, *, row_budget: int) -> np.ndarray:
    if y.shape[1] > row_budget:
        raise RuntimeError(f"Row count N={y.shape[1]} exceeds row_budget={row_budget}.")
    if y.shape[1] == row_budget:
        return y
    pad_rows = row_budget - y.shape[1]
    return np.pad(y, ((0, 0), (0, pad_rows)), mode="constant")
