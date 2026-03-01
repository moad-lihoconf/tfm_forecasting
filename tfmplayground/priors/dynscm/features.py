"""Forecast featurization for DynSCM."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .config import DynSCMConfig


def sample_origins_and_horizons(
    cfg: DynSCMConfig,
    *,
    batch_size: int,
    num_steps: int,
    n_train: int,
    n_test: int,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample origin and horizon indices with strict train/test chronology."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")
    if num_steps < 2:
        raise ValueError("num_steps must be >= 2.")
    if n_train < 1 or n_test < 1:
        raise ValueError("n_train and n_test must both be >= 1.")

    generator = cfg.make_rng(seed) if rng is None else rng
    num_rows = n_train + n_test

    horizons = np.asarray(cfg.forecast_horizons, dtype=np.int64)
    if horizons.ndim != 1 or horizons.size == 0:
        raise ValueError("forecast_horizons must be a non-empty 1D sequence.")
    if np.any(horizons <= 0):
        raise ValueError("forecast_horizons must contain positive integers.")

    h_idx = generator.choice(horizons, size=(batch_size, num_rows), replace=True)

    required_lag = max(cfg.max_feature_lag, int(np.max(cfg.explicit_lags)))
    max_horizon = int(np.max(horizons))
    min_origin = required_lag
    max_origin = num_steps - 1 - max_horizon
    if max_origin <= min_origin:
        raise ValueError(
            f"Need max_origin > min_origin; got {max_origin=} and {min_origin=}."
        )

    split_points = generator.integers(min_origin, max_origin, size=(batch_size,))
    train_low = np.full((batch_size,), min_origin, dtype=np.int64)
    train_high = split_points
    test_low = split_points + 1
    test_high = np.full((batch_size,), max_origin, dtype=np.int64)

    t_train = _sample_int_ranges(
        generator=generator,
        low=train_low,
        high=train_high,
        width=n_train,
    )
    t_test = _sample_int_ranges(
        generator=generator,
        low=test_low,
        high=test_high,
        width=n_test,
    )
    t_idx = np.concatenate([t_train, t_test], axis=1).astype(np.int64)

    _validate_time_layout(
        t_idx=t_idx,
        h_idx=h_idx,
        num_steps=num_steps,
        n_train=n_train,
        required_lag=required_lag,
    )
    return t_idx, h_idx.astype(np.int64)


def build_forecasting_table(
    cfg: DynSCMConfig,
    series: np.ndarray,
    y_idx: np.ndarray,
    *,
    n_train: int,
    n_test: int,
    t_idx: np.ndarray | None = None,
    h_idx: np.ndarray | None = None,
    obs_mask: np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build forecast table `(x, y)` and metadata from raw series."""
    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("series must have shape (B, T, p).")
    batch_size, num_steps, num_vars = values.shape

    y_index = np.asarray(y_idx, dtype=np.int64)
    if y_index.shape != (batch_size,):
        raise ValueError("y_idx must have shape (B,).")
    if y_index.min() < 0 or y_index.max() >= num_vars:
        raise ValueError("y_idx values must be in [0, p).")

    generator = cfg.make_rng(seed)
    if t_idx is None or h_idx is None:
        sampled_t, sampled_h = sample_origins_and_horizons(
            cfg,
            batch_size=batch_size,
            num_steps=num_steps,
            n_train=n_train,
            n_test=n_test,
            rng=generator,
        )
        t_idx = sampled_t if t_idx is None else np.asarray(t_idx, dtype=np.int64)
        h_idx = sampled_h if h_idx is None else np.asarray(h_idx, dtype=np.int64)
    else:
        t_idx = np.asarray(t_idx, dtype=np.int64)
        h_idx = np.asarray(h_idx, dtype=np.int64)

    required_lag = max(cfg.max_feature_lag, int(np.max(cfg.explicit_lags)))
    _validate_time_layout(
        t_idx=t_idx,
        h_idx=h_idx,
        num_steps=num_steps,
        n_train=n_train,
        required_lag=required_lag,
    )
    num_rows = t_idx.shape[1]
    if num_rows != n_train + n_test:
        raise ValueError("t_idx/h_idx row count must equal n_train + n_test.")

    if obs_mask is None:
        observed_mask = np.ones((batch_size, num_steps, num_vars), dtype=bool)
    else:
        observed_mask = np.asarray(obs_mask, dtype=bool)
        if observed_mask.shape != (batch_size, num_steps, num_vars):
            raise ValueError("obs_mask must have shape (B, T, p).")

    lag_values, lag_observed = _extract_lag_features(
        values=values,
        observed_mask=observed_mask,
        t_idx=t_idx,
        lags=np.asarray(cfg.explicit_lags, dtype=np.int64),
    )
    kernels = _sample_kernels(cfg=cfg, rng=generator)
    kernel_values, kernel_observed = _extract_kernel_features(
        values=values,
        observed_mask=observed_mask,
        t_idx=t_idx,
        kernels=kernels,
    )

    data_values = _concat_feature_blocks([lag_values, kernel_values])
    data_observed = _concat_feature_blocks([lag_observed, kernel_observed]).astype(bool)
    imputed_data = _impute_with_train_means(
        values=data_values,
        observed=data_observed,
        n_train=n_train,
    )

    deterministic_values, deterministic_slices = _build_deterministic_channels(
        cfg=cfg,
        t_idx=t_idx,
        h_idx=h_idx,
        num_steps=num_steps,
        rng=generator,
    )

    include_mask_channels = bool(cfg.add_mask_channels)
    if cfg.disable_mask_channels_when_missing_off and cfg.missing_mode == "off":
        include_mask_channels = False

    x_parts: list[tuple[str, np.ndarray]] = [("data_values", imputed_data)]
    if include_mask_channels and data_observed.shape[2] > 0:
        x_parts.append(("data_mask", data_observed.astype(np.float64)))
    if deterministic_values.shape[2] > 0:
        x_parts.append(("deterministic", deterministic_values))

    x, feature_slices = _concat_named_blocks(x_parts)

    # Add lags_value / kern_value sub-slices within data_values.
    if "data_values" in feature_slices:
        dv_start, _ = feature_slices["data_values"]
        f_lag = int(lag_values.shape[2])
        f_ker = int(kernel_values.shape[2])
        feature_slices["lags_value"] = (dv_start, dv_start + f_lag)
        feature_slices["kern_value"] = (dv_start + f_lag, dv_start + f_lag + f_ker)

    # Merge deterministic sub-slices into global coordinates.
    if "deterministic" in feature_slices:
        det_start, _ = feature_slices["deterministic"]
        for key, (local_start, local_end) in deterministic_slices.items():
            feature_slices[key] = (det_start + local_start, det_start + local_end)

    y_time = t_idx + h_idx
    batch_index = np.arange(batch_size, dtype=np.int64)[:, None]
    y = values[batch_index, y_time, y_index[:, None]]

    if not np.isfinite(x).all():
        raise RuntimeError("Feature matrix contains non-finite values.")
    if not np.isfinite(y).all():
        raise RuntimeError("Label matrix contains non-finite values.")

    metadata: dict[str, Any] = {
        "t_idx": t_idx.astype(np.int64),
        "h_idx": h_idx.astype(np.int64),
        "feature_slices": feature_slices,
        "F_used": int(x.shape[2]),
        "single_eval_pos": int(n_train),
    }
    return x.astype(np.float64), y.astype(np.float64), metadata


def _sample_int_ranges(
    *,
    generator: np.random.Generator,
    low: np.ndarray,
    high: np.ndarray,
    width: int,
) -> np.ndarray:
    """Sample integers independently from inclusive per-batch ranges."""
    if width < 1:
        raise ValueError("width must be >= 1.")
    if low.shape != high.shape:
        raise ValueError("low and high must have identical shapes.")
    if np.any(high < low):
        raise ValueError("Each high endpoint must be >= corresponding low endpoint.")

    span = (high - low + 1).astype(np.int64)
    max_span = int(np.max(span))
    # NOTE: modulo mapping introduces small bias when max_span is not a
    # multiple of span[b].  Acceptable for prior sampling; revisit if exact
    # uniformity is needed.
    raw = generator.integers(0, max_span, size=(low.shape[0], width))
    sampled = low[:, None] + (raw % span[:, None])
    return sampled.astype(np.int64)


def _validate_time_layout(
    *,
    t_idx: np.ndarray,
    h_idx: np.ndarray,
    num_steps: int,
    n_train: int,
    required_lag: int,
) -> None:
    if t_idx.shape != h_idx.shape:
        raise ValueError("t_idx and h_idx must have identical shape.")
    if t_idx.ndim != 2:
        raise ValueError("t_idx and h_idx must have shape (B, N).")
    if t_idx.shape[1] <= n_train:
        raise ValueError("Need at least one test row: N must be > n_train.")
    if n_train < 1:
        raise ValueError("n_train must be >= 1.")

    if np.any(t_idx < required_lag):
        raise ValueError("t_idx must satisfy t_idx >= max_feature_lag safety floor.")
    if np.any(h_idx < 1):
        raise ValueError("h_idx must contain positive horizons.")
    if np.any(t_idx + h_idx >= num_steps):
        raise ValueError("All label times must satisfy t_idx + h_idx < T.")

    train_max = np.max(t_idx[:, :n_train], axis=1)
    test_min = np.min(t_idx[:, n_train:], axis=1)
    if np.any(train_max >= test_min):
        raise ValueError("Strict chronology violated: max(train) must be < min(test).")


def _extract_lag_features(
    *,
    values: np.ndarray,
    observed_mask: np.ndarray,
    t_idx: np.ndarray,
    lags: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size, _, num_vars = values.shape
    num_rows = t_idx.shape[1]
    num_lags = lags.size
    if num_lags == 0:
        empty = np.zeros((batch_size, num_rows, 0), dtype=np.float64)
        return empty, empty.astype(bool)

    lag_times = t_idx[:, :, None] - lags[None, None, :]
    batch_index = np.arange(batch_size, dtype=np.int64)[:, None, None, None]
    var_index = np.arange(num_vars, dtype=np.int64)[None, None, None, :]
    lag_values = values[batch_index, lag_times[:, :, :, None], var_index]
    lag_obs = observed_mask[batch_index, lag_times[:, :, :, None], var_index]

    lag_values = lag_values.reshape(batch_size, num_rows, num_lags * num_vars)
    lag_obs = lag_obs.reshape(batch_size, num_rows, num_lags * num_vars)
    return lag_values.astype(np.float64), lag_obs.astype(bool)


def _sample_kernels(
    *,
    cfg: DynSCMConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    num_kernels = cfg.num_kernels
    max_lag = cfg.max_feature_lag
    if num_kernels <= 0:
        return np.zeros((0, max_lag), dtype=np.float64)

    lag_axis = np.arange(1, max_lag + 1, dtype=np.float64)
    kernels = np.zeros((num_kernels, max_lag), dtype=np.float64)
    family = str(cfg.kernel_family)

    for kernel_idx in range(num_kernels):
        if family == "exp_decay":
            alpha = float(rng.uniform(0.05, 0.35))
            kernel = np.exp(-alpha * lag_axis)
        elif family == "power_law":
            beta = float(rng.uniform(0.6, 1.8))
            kernel = lag_axis ** (-beta)
        elif family == "mix":
            if kernel_idx == 0:
                # Short-memory kernel: fixed fast exp-decay.
                kernel = np.exp(-0.8 * lag_axis)
            elif kernel_idx == 1:
                # Long-memory kernel: fixed slow power-law.
                kernel = lag_axis ** (-1.0)
            else:
                # Random convex combination of exp-decay and power-law.
                a = float(rng.uniform(0.0, 1.0))
                alpha = float(rng.uniform(0.05, 0.35))
                beta = float(rng.uniform(0.6, 1.8))
                kernel = a * np.exp(-alpha * lag_axis) + (1.0 - a) * lag_axis ** (-beta)
        else:
            raise ValueError(f"Unsupported kernel_family={family!r}.")

        kernel = np.clip(kernel, 0.0, None)
        kernel_sum = float(np.sum(kernel))
        kernels[kernel_idx] = kernel / kernel_sum if kernel_sum > 0.0 else 0.0

    return kernels


def _extract_kernel_features(
    *,
    values: np.ndarray,
    observed_mask: np.ndarray,
    t_idx: np.ndarray,
    kernels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    batch_size, num_steps, num_vars = values.shape
    num_rows = t_idx.shape[1]
    num_kernels = kernels.shape[0]
    max_lag = kernels.shape[1] if num_kernels > 0 else 0
    if num_kernels == 0:
        empty = np.zeros((batch_size, num_rows, 0), dtype=np.float64)
        return empty, empty.astype(bool)

    if max_lag < 1:
        raise ValueError("Kernel lag length must be >= 1.")
    if np.any(t_idx < max_lag):
        raise ValueError("Kernel summaries require t_idx >= max_feature_lag.")

    windows = sliding_window_view(values, window_shape=max_lag, axis=1)
    mask_windows = sliding_window_view(
        observed_mask.astype(np.float64),
        max_lag,
        axis=1,
    )
    gather_idx = t_idx - max_lag
    batch_index = np.arange(batch_size, dtype=np.int64)[:, None]

    gathered_values = windows[batch_index, gather_idx]  # (B, N, p, L)
    gathered_mask = mask_windows[batch_index, gather_idx]  # (B, N, p, L)

    numerator = np.einsum("bnpl,ml->bnpm", gathered_values, kernels, optimize=True)
    denominator = np.einsum("bnpl,ml->bnpm", gathered_mask, kernels, optimize=True)
    observed = denominator > 1e-8
    kernel_values = np.where(observed, numerator / np.maximum(denominator, 1e-8), 0.0)

    kernel_values = kernel_values.reshape(batch_size, num_rows, num_vars * num_kernels)
    observed = observed.reshape(batch_size, num_rows, num_vars * num_kernels)
    return kernel_values.astype(np.float64), observed.astype(bool)


def _impute_with_train_means(
    *,
    values: np.ndarray,
    observed: np.ndarray,
    n_train: int,
) -> np.ndarray:
    if values.shape != observed.shape:
        raise ValueError("values and observed must have identical shape.")
    if values.shape[2] == 0:
        return values.astype(np.float64, copy=True)

    train_values = values[:, :n_train, :]
    train_observed = observed[:, :n_train, :].astype(np.float64)
    observed_sum = np.sum(train_values * train_observed, axis=1)
    observed_count = np.sum(train_observed, axis=1)

    train_means = np.zeros_like(observed_sum)
    valid = observed_count > 0.0
    train_means[valid] = observed_sum[valid] / observed_count[valid]

    return np.where(observed, values, train_means[:, None, :]).astype(np.float64)


def _build_deterministic_channels(
    *,
    cfg: DynSCMConfig,
    t_idx: np.ndarray,
    h_idx: np.ndarray,
    num_steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    blocks: list[tuple[str, np.ndarray]] = []

    if cfg.add_time_feature:
        denom = float(max(num_steps - 1, 1))
        blocks.append(("time", (t_idx.astype(np.float64) / denom)[:, :, None]))

    if cfg.add_horizon_feature:
        horizon_scale = float(max(np.max(cfg.forecast_horizons), 1))
        horizon = (h_idx.astype(np.float64) / horizon_scale)[:, :, None]
        blocks.append(("horizon", horizon))
        if cfg.add_log_horizon:
            log_horizon = np.log1p(h_idx.astype(np.float64))[:, :, None]
            blocks.append(("log_horizon", log_horizon))

    if cfg.add_seasonality:
        periods = rng.choice(
            np.asarray(cfg.season_period_choices, dtype=np.float64),
            size=(t_idx.shape[0],),
            replace=True,
        )
        phase = (2.0 * np.pi) * t_idx.astype(np.float64) / periods[:, None]
        seasonality = np.stack([np.sin(phase), np.cos(phase)], axis=-1)
        blocks.append(("seasonality", seasonality))

    if not blocks:
        empty = np.zeros((t_idx.shape[0], t_idx.shape[1], 0), dtype=np.float64)
        return empty, {}
    return _concat_named_blocks(blocks)


def _concat_feature_blocks(blocks: list[np.ndarray]) -> np.ndarray:
    if not blocks:
        return np.zeros((0, 0, 0), dtype=np.float64)
    if len(blocks) == 1:
        return blocks[0]
    return np.concatenate(blocks, axis=2)


def _concat_named_blocks(
    blocks: list[tuple[str, np.ndarray]],
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    if not blocks:
        raise ValueError("Named blocks list must be non-empty.")

    names: list[str] = []
    arrays: list[np.ndarray] = []
    for name, array in blocks:
        if array.ndim != 3:
            raise ValueError(f"Block {name!r} must have shape (B, N, F).")
        names.append(name)
        arrays.append(array.astype(np.float64))

    output = np.concatenate(arrays, axis=2)
    slices: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, array in zip(names, arrays, strict=True):
        width = int(array.shape[2])
        slices[name] = (cursor, cursor + width)
        cursor += width
    return output, slices


def extract_feature_block(
    x: np.ndarray,
    metadata: Mapping[str, Any],
    block_name: str,
) -> np.ndarray:
    """Convenience helper to read a block from `x` using metadata slices."""
    slices = metadata.get("feature_slices")
    if not isinstance(slices, Mapping):
        raise ValueError("metadata['feature_slices'] must be a mapping.")
    if block_name not in slices:
        raise KeyError(f"Unknown block {block_name!r}.")
    start, end = slices[block_name]
    return x[:, :, int(start) : int(end)]
