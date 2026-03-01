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

from .config import DynSCMConfig, sample_dynscm_variant_cfg
from .difficulty import measure_sample_learnability
from .features import build_forecasting_table, sample_origins_and_horizons
from .graph import sample_regime_graphs
from .mechanisms import sample_regime_mechanisms
from .missingness import sample_observation_mask
from .research import DynSCMSampleFilterConfig
from .simulate import simulate_dynscm_series
from .stability import sample_stable_coefficients

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
    cfg_overrides: dict[str, object] | None = None
    filter_payload: dict[str, object] | None = None
    max_generation_attempts: int = 1


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
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float | str]]:
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
        cfg_overrides=task.cfg_overrides,
        sample_filter=DynSCMSampleFilterConfig.from_payload(task.filter_payload),
        max_generation_attempts=task.max_generation_attempts,
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
    cfg_overrides: Mapping[str, object] | None = None,
    sample_filter: DynSCMSampleFilterConfig | None = None,
    max_generation_attempts: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float | str]]:
    """Generate exactly one padded DynSCM `(x_i, y_i)` sample."""
    if max_generation_attempts < 1:
        raise ValueError("max_generation_attempts must be >= 1.")
    base_cfg = cfg.with_overrides(**dict(cfg_overrides)) if cfg_overrides else cfg
    last_sample: tuple[np.ndarray, np.ndarray, dict[str, int | float | str]] | None = (
        None
    )
    last_exc: RuntimeError | None = None
    reject_counts = {
        "low_std": 0,
        "probe_r2_low": 0,
        "probe_r2_high": 0,
        "clipped": 0,
        "max_abs_value": 0,
        "informative_features_low": 0,
        "missing_fraction_high": 0,
        "block_missing_fraction_high": 0,
    }
    attempt_count = 0
    for attempt_idx in range(max_generation_attempts):
        attempt_seed = _derive_attempt_seed(sample_seed, attempt_idx)
        attempt_count = attempt_idx + 1
        try:
            x_padded, y_padded, sample_metadata = _build_single_dynscm_sample_once(
                base_cfg,
                sample_seed=int(attempt_seed),
                n_train=n_train,
                n_test=n_test,
                row_budget=row_budget,
                num_features=num_features,
            )
        except RuntimeError as exc:
            last_exc = exc
            continue
        rejection_reason = (
            sample_filter.rejection_reason(sample_metadata) if sample_filter else None
        )
        accepted = rejection_reason is None
        sample_metadata["sampled_filter_accept"] = int(accepted)
        if rejection_reason is not None:
            reject_counts[rejection_reason] = reject_counts.get(rejection_reason, 0) + 1
        last_sample = (x_padded, y_padded, sample_metadata)
        if accepted:
            break
    if last_sample is None:
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("DynSCM sample generation failed to produce any sample.")
    last_sample[2]["sampled_generation_attempts_used"] = int(attempt_count)
    last_sample[2]["sampled_low_std_reject_count"] = int(reject_counts["low_std"])
    last_sample[2]["sampled_probe_r2_reject_count"] = int(
        reject_counts["probe_r2_low"] + reject_counts["probe_r2_high"]
    )
    last_sample[2]["sampled_clipped_reject_count"] = int(reject_counts["clipped"])
    last_sample[2]["sampled_max_abs_reject_count"] = int(reject_counts["max_abs_value"])
    last_sample[2]["sampled_informative_feature_reject_count"] = int(
        reject_counts["informative_features_low"]
    )
    last_sample[2]["sampled_missing_reject_count"] = int(
        reject_counts["missing_fraction_high"]
        + reject_counts["block_missing_fraction_high"]
    )
    return last_sample


def _derive_attempt_seed(sample_seed: int, attempt_idx: int) -> int:
    seed_seq = np.random.SeedSequence(
        [int(sample_seed) % (2**32), int(attempt_idx) % (2**32)]
    )
    return int(seed_seq.generate_state(1, dtype=np.uint64)[0] % _SEED_MAX)


def _sample_train_target_std(y_raw: np.ndarray, n_train: int) -> float:
    train_targets = np.asarray(y_raw[0, :n_train], dtype=np.float64)
    if train_targets.size <= 1:
        return 0.0
    return float(np.std(train_targets, ddof=1))


def _target_scale_risk(max_abs_target_value: float, train_target_std: float) -> int:
    if max_abs_target_value > 1000.0:
        return 1
    if train_target_std > 0.0 and max_abs_target_value > (1000.0 * train_target_std):
        return 1
    return 0


def _build_single_dynscm_sample_once(
    cfg: DynSCMConfig,
    *,
    sample_seed: int,
    n_train: int,
    n_test: int,
    row_budget: int,
    num_features: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float | str]]:
    sample_rng = cfg.make_rng(sample_seed)
    variant_cfg, variant_metadata = sample_dynscm_variant_cfg(cfg, sample_rng)
    num_vars, num_steps, y_idx = sample_dataset_dimensions(variant_cfg, sample_rng)
    per_sample_seeds = sample_rng.integers(
        0,
        _SEED_MAX,
        size=(7,),
        dtype=np.int64,
    )

    graph_sample = sample_regime_graphs(
        variant_cfg,
        num_vars=num_vars,
        target_idx=y_idx,
        seed=int(per_sample_seeds[0]),
    )
    stability_sample = sample_stable_coefficients(
        variant_cfg,
        graph_sample,
        target_idx=y_idx,
        seed=int(per_sample_seeds[1]),
    )
    mechanism_sample = sample_regime_mechanisms(
        variant_cfg,
        graph_sample,
        stability_sample=stability_sample,
        target_idx=y_idx,
        seed=int(per_sample_seeds[2]),
    )
    simulation_sample = simulate_dynscm_series(
        variant_cfg,
        graph_sample,
        mechanism_sample,
        num_steps=num_steps,
        seed=int(per_sample_seeds[3]),
    )

    series = simulation_sample.series[None, :, :].astype(np.float64, copy=False)
    y_index = np.array([y_idx], dtype=np.int64)

    t_idx, h_idx = sample_origins_and_horizons(
        variant_cfg,
        batch_size=1,
        num_steps=num_steps,
        n_train=n_train,
        n_test=n_test,
        seed=int(per_sample_seeds[4]),
    )
    obs_mask = sample_observation_mask(
        variant_cfg,
        series,
        seed=int(per_sample_seeds[5]),
        label_times=t_idx + h_idx,
        label_var_indices=y_index,
    )

    x_raw, y_raw, metadata = build_forecasting_table(
        variant_cfg,
        series,
        y_index,
        n_train=n_train,
        n_test=n_test,
        t_idx=t_idx,
        h_idx=h_idx,
        obs_mask=obs_mask,
        seed=int(per_sample_seeds[6]),
    )

    x_prioritized = prioritize_feature_blocks(
        x_raw,
        feature_slices=metadata["feature_slices"],
    )
    x_budgeted = fit_feature_budget(x_prioritized, num_features=num_features)
    x_padded = pad_rows_3d(x_budgeted, row_budget=row_budget)
    y_padded = pad_rows_2d(y_raw, row_budget=row_budget)
    learnability = measure_sample_learnability(
        x=x_prioritized[0],
        y=y_raw[0],
        n_train=n_train,
        observed_mask=obs_mask,
        informative_feature_std_floor=float(
            variant_cfg.informative_feature_std_floor
            if variant_cfg.informative_feature_std_floor is not None
            else 1e-3
        ),
        compute_probe_r2=bool(variant_cfg.learnability_probe),
        has_long_history=bool(
            variant_cfg.series_length_max > stable_length_baseline(variant_cfg)
        ),
        has_regimes=bool(variant_cfg.num_regimes > 1),
        has_drift=bool(variant_cfg.drift_std > 0.0),
        has_missingness=bool(variant_cfg.missing_mode != "off"),
        has_heavy_tails=bool(variant_cfg.noise_family == "student_t"),
    )
    data_mask_slice = metadata["feature_slices"].get("data_mask")
    mask_channels_enabled = int(
        data_mask_slice is not None
        and int(data_mask_slice[1]) > int(data_mask_slice[0])
    )
    min_target_parent_count = int(
        np.min(graph_sample.target_lag_parent_counts)
        if np.all(graph_sample.target_lag_parent_counts >= 0)
        else 0
    )
    min_target_self_lag_weight = float(
        np.min(stability_sample.target_self_lag_weights)
        if stability_sample.target_self_lag_weights.size > 0
        else 0.0
    )

    sample_metadata = {
        **variant_metadata,
        "sampled_num_vars": int(num_vars),
        "sampled_num_steps": int(num_steps),
        "sampled_n_train": int(n_train),
        "sampled_n_test": int(n_test),
        "sampled_pre_budget_feature_count": int(x_prioritized.shape[2]),
        "sampled_simulation_clipped": int(bool(simulation_sample.clipped)),
        "sampled_simulation_num_attempts": int(simulation_sample.num_attempts),
        "sampled_simulation_max_abs_value": float(simulation_sample.max_abs_value),
        "sampled_train_target_std": float(learnability.train_target_std),
        "sampled_test_target_std": float(learnability.test_target_std),
        "sampled_max_abs_target_value": float(learnability.max_abs_target_value),
        "sampled_probe_r2": (
            float(learnability.probe_r2) if learnability.probe_r2 is not None else 0.0
        ),
        "sampled_informative_feature_count": int(
            learnability.informative_feature_count
        ),
        "sampled_informative_feature_std_floor": float(
            learnability.informative_feature_std_floor
        ),
        "sampled_target_parent_count": int(min_target_parent_count),
        "sampled_target_self_lag_weight": float(min_target_self_lag_weight),
        "sampled_target_had_forced_lag_parent": int(
            bool(np.any(graph_sample.forced_target_lag_parent))
        ),
        "sampled_target_had_forced_self_lag": int(
            bool(
                np.any(graph_sample.forced_target_self_lag)
                or np.any(stability_sample.forced_target_self_lag)
            )
        ),
        "sampled_mask_channels_enabled": int(mask_channels_enabled),
        "sampled_noise_scale": float(simulation_sample.noise_scales[int(y_idx)]),
        "sampled_missing_fraction": float(learnability.missing_fraction),
        "sampled_block_missing_fraction": float(learnability.block_missing_fraction),
        "sampled_target_scale_high_risk": int(
            _target_scale_risk(
                max_abs_target_value=learnability.max_abs_target_value,
                train_target_std=learnability.train_target_std,
            )
        ),
        "sampled_filter_accept": 1,
    }

    return (
        x_padded[0].astype(np.float32, copy=False),
        y_padded[0].astype(np.float32, copy=False),
        sample_metadata,
    )


def stable_length_baseline(cfg: DynSCMConfig) -> int:
    return max(int(cfg.train_rows_max) + int(cfg.test_rows_max), 128)


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
