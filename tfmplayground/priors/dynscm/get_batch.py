"""Batch assembly for DynSCM forecasting prior samples."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import torch

from .config import DynSCMConfig
from .features import build_forecasting_table, sample_origins_and_horizons
from .graph import sample_regime_graphs
from .mechanisms import sample_regime_mechanisms
from .missingness import sample_observation_mask
from .simulate import simulate_dynscm_series


def make_get_batch_dynscm(
    cfg: DynSCMConfig,
    device: torch.device,
    seed: int | None = None,
) -> Callable[[int, int, int], dict[str, torch.Tensor | int]]:
    """Return a stateful `get_batch` closure compatible with `PriorDataLoader`.

    The closure keeps an internal RNG that advances per call, making successive
    batches deterministic for a fixed `(cfg, seed)` pair.
    """
    target_device = torch.device(device)
    state_rng = cfg.make_rng(seed)

    def get_batch(
        batch_size: int,
        num_datapoints_max: int,
        num_features: int,
    ) -> dict[str, torch.Tensor | int]:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if num_datapoints_max < 2:
            raise ValueError("num_datapoints_max must be >= 2.")
        if num_features < 1:
            raise ValueError("num_features must be >= 1.")

        n_train, n_test = _sample_shared_row_counts(
            cfg=cfg,
            num_datapoints_max=num_datapoints_max,
            rng=state_rng,
        )

        x_items: list[np.ndarray] = []
        y_items: list[np.ndarray] = []

        for _ in range(batch_size):
            num_vars, num_steps, y_idx = _sample_dataset_dimensions(cfg, state_rng)

            graph_sample = sample_regime_graphs(
                cfg,
                num_vars=num_vars,
                seed=_draw_seed(state_rng),
            )
            mechanism_sample = sample_regime_mechanisms(
                cfg,
                graph_sample,
                seed=_draw_seed(state_rng),
            )
            simulation_sample = simulate_dynscm_series(
                cfg,
                graph_sample,
                mechanism_sample,
                num_steps=num_steps,
                seed=_draw_seed(state_rng),
            )

            series = simulation_sample.series[None, :, :].astype(np.float64, copy=False)
            y_index = np.array([y_idx], dtype=np.int64)

            t_idx, h_idx = sample_origins_and_horizons(
                cfg,
                batch_size=1,
                num_steps=num_steps,
                n_train=n_train,
                n_test=n_test,
                seed=_draw_seed(state_rng),
            )
            obs_mask = sample_observation_mask(
                cfg,
                series,
                seed=_draw_seed(state_rng),
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
                seed=_draw_seed(state_rng),
            )

            x_prioritized = _prioritize_feature_blocks(
                x_raw,
                feature_slices=metadata["feature_slices"],
            )
            x_budgeted = _fit_feature_budget(x_prioritized, num_features=num_features)
            x_padded = _pad_rows_3d(x_budgeted, row_budget=num_datapoints_max)
            y_padded = _pad_rows_2d(y_raw, row_budget=num_datapoints_max)

            x_items.append(x_padded[0])
            y_items.append(y_padded[0])

        x_batch = np.stack(x_items, axis=0).astype(np.float32, copy=False)
        y_batch = np.stack(y_items, axis=0).astype(np.float32, copy=False)

        if not np.isfinite(x_batch).all():
            raise RuntimeError(
                "DynSCM batch feature tensor contains non-finite values."
            )
        if not np.isfinite(y_batch).all():
            raise RuntimeError("DynSCM batch label tensor contains non-finite values.")

        x_tensor = torch.from_numpy(x_batch).to(target_device)
        y_tensor = torch.from_numpy(y_batch).to(target_device)

        return {
            "x": x_tensor,
            "y": y_tensor,
            "target_y": y_tensor.clone(),
            "single_eval_pos": int(n_train),
        }

    return get_batch


def _sample_shared_row_counts(
    *,
    cfg: DynSCMConfig,
    num_datapoints_max: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    train_min = max(1, cfg.train_rows_min)
    train_max = max(train_min, cfg.train_rows_max)
    test_min = max(1, cfg.test_rows_min)
    test_max = max(test_min, cfg.test_rows_max)

    # O(train_range × test_range) enumeration; ~500 pairs with default config
    # bounds.  Acceptable for prior sampling; consider caching if profiling
    # shows overhead.
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

    pair_idx = int(rng.integers(0, len(feasible_pairs)))
    return feasible_pairs[pair_idx]


def _required_min_series_length(cfg: DynSCMConfig) -> int:
    required_lag = max(cfg.max_feature_lag, max(cfg.explicit_lags))
    max_horizon = int(max(cfg.forecast_horizons))
    return required_lag + max_horizon + 2


def _sample_dataset_dimensions(
    cfg: DynSCMConfig,
    rng: np.random.Generator,
) -> tuple[int, int, int]:
    num_vars = int(
        rng.integers(
            cfg.num_variables_min,
            cfg.num_variables_max + 1,
        )
    )

    series_len_min = max(cfg.series_length_min, _required_min_series_length(cfg))
    series_len_max = cfg.series_length_max
    if series_len_min > series_len_max:
        raise ValueError(
            "series_length_max is too small for feature/horizon safety: "
            f"need at least {series_len_min}, got {series_len_max}."
        )
    num_steps = int(rng.integers(series_len_min, series_len_max + 1))

    y_idx = int(rng.integers(0, num_vars))
    return num_vars, num_steps, y_idx


def _slice_or_empty(
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


def _prioritize_feature_blocks(
    x: np.ndarray,
    *,
    feature_slices: Mapping[str, tuple[int, int]],
) -> np.ndarray:
    """Return features ordered by deterministic priority for truncation safety.

    Priority order:
    1) deterministic channels (time/horizon/seasonality)
    2) explicit lag values
    3) kernel summary values
    4) data mask channels
    """
    blocks = [
        _slice_or_empty(x, feature_slices, "deterministic"),
        _slice_or_empty(x, feature_slices, "lags_value"),
        _slice_or_empty(x, feature_slices, "kern_value"),
        _slice_or_empty(x, feature_slices, "data_mask"),
    ]
    non_empty = [block for block in blocks if block.shape[2] > 0]
    if not non_empty:
        return np.zeros((x.shape[0], x.shape[1], 0), dtype=x.dtype)
    return np.concatenate(non_empty, axis=2)


def _fit_feature_budget(x: np.ndarray, *, num_features: int) -> np.ndarray:
    if x.shape[2] >= num_features:
        return x[:, :, :num_features]
    pad_width = num_features - x.shape[2]
    return np.pad(x, ((0, 0), (0, 0), (0, pad_width)), mode="constant")


def _pad_rows_3d(x: np.ndarray, *, row_budget: int) -> np.ndarray:
    if x.shape[1] > row_budget:
        raise RuntimeError(f"Row count N={x.shape[1]} exceeds row_budget={row_budget}.")
    if x.shape[1] == row_budget:
        return x
    pad_rows = row_budget - x.shape[1]
    return np.pad(x, ((0, 0), (0, pad_rows), (0, 0)), mode="constant")


def _pad_rows_2d(y: np.ndarray, *, row_budget: int) -> np.ndarray:
    if y.shape[1] > row_budget:
        raise RuntimeError(f"Row count N={y.shape[1]} exceeds row_budget={row_budget}.")
    if y.shape[1] == row_budget:
        return y
    pad_rows = row_budget - y.shape[1]
    return np.pad(y, ((0, 0), (0, pad_rows)), mode="constant")


def _draw_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, np.iinfo(np.int64).max))
