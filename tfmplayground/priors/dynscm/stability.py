"""Stability guard utilities for DynSCM coefficient sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DynSCMConfig
from .graph import DynSCMGraphSample


@dataclass(frozen=True, slots=True)
class DynSCMStabilitySample:
    """Sampled stable coefficients coupled to a DynSCM regime graph."""

    num_vars: int
    num_regimes: int
    max_lag: int
    lag_coefficients: np.ndarray  # (K, L, p, p), source->target coefficients.
    contemporaneous_coefficients: np.ndarray  # (K, p, p), source->target coefficients.
    lag_column_budgets: np.ndarray  # (K, p), per-target budgets for lag edges.
    contemp_column_budgets: np.ndarray  # (K, p), per-target budgets for cont. edges.
    guardc_scale_factors: np.ndarray  # (K,), multiplicative scale applied to lag block.
    lag_spectral_radii: np.ndarray  # (K,), spectral radius after optional guard-c.


def sample_stable_coefficients(
    cfg: DynSCMConfig,
    graph_sample: DynSCMGraphSample,
    *,
    seed: int | None = None,
) -> DynSCMStabilitySample:
    """Sample regime-specific coefficients with Guard A and optional Guard C."""
    num_regimes = graph_sample.num_regimes
    max_lag = graph_sample.max_lag
    num_vars = graph_sample.num_vars

    if cfg.max_lag != max_lag:
        raise ValueError(
            f"Graph max_lag={max_lag} does not match config max_lag={cfg.max_lag}."
        )
    if cfg.num_regimes != num_regimes:
        raise ValueError(
            "Graph num_regimes="
            f"{num_regimes} does not match config num_regimes={cfg.num_regimes}."
        )

    rng = cfg.make_rng(seed)

    lag_coefficients = np.zeros(
        (num_regimes, max_lag, num_vars, num_vars), dtype=np.float64
    )
    contemporaneous_coefficients = np.zeros(
        (num_regimes, num_vars, num_vars), dtype=np.float64
    )
    lag_column_budgets = np.zeros((num_regimes, num_vars), dtype=np.float64)
    contemp_column_budgets = np.zeros((num_regimes, num_vars), dtype=np.float64)
    guardc_scale_factors = np.ones((num_regimes,), dtype=np.float64)
    lag_spectral_radii = np.zeros((num_regimes,), dtype=np.float64)

    for regime_idx in range(num_regimes):
        lag_budget = rng.uniform(
            cfg.col_budget_min, cfg.col_budget_max, size=(num_vars,)
        ).astype(np.float64)
        cont_budget = rng.uniform(0.0, cfg.contemp_budget_max, size=(num_vars,)).astype(
            np.float64
        )

        lag_mask = graph_sample.regime_lagged_adjacency[regime_idx]
        cont_mask = graph_sample.regime_contemporaneous_adjacency[regime_idx]

        lag_block = sample_signed_budgeted_coefficients(
            edge_mask=lag_mask,
            target_budgets=lag_budget,
            rng=rng,
        )
        cont_block = sample_signed_budgeted_coefficients(
            edge_mask=cont_mask[None, ...],
            target_budgets=cont_budget,
            rng=rng,
        )[0]

        if cfg.enable_guardc_rescale:
            lag_block, scale, rho = guardc_rescale_lag_block(
                lag_coefficients=lag_block,
                delta=cfg.guardc_delta,
            )
            guardc_scale_factors[regime_idx] = scale
            lag_spectral_radii[regime_idx] = rho
        else:
            lag_spectral_radii[regime_idx] = companion_spectral_radius(lag_block)

        lag_coefficients[regime_idx] = lag_block
        contemporaneous_coefficients[regime_idx] = cont_block
        lag_column_budgets[regime_idx] = lag_budget
        contemp_column_budgets[regime_idx] = cont_budget

    _validate_stability_sample(
        cfg=cfg,
        graph_sample=graph_sample,
        lag_coefficients=lag_coefficients,
        contemporaneous_coefficients=contemporaneous_coefficients,
        lag_column_budgets=lag_column_budgets,
        contemp_column_budgets=contemp_column_budgets,
        lag_spectral_radii=lag_spectral_radii,
    )

    return DynSCMStabilitySample(
        num_vars=num_vars,
        num_regimes=num_regimes,
        max_lag=max_lag,
        lag_coefficients=lag_coefficients,
        contemporaneous_coefficients=contemporaneous_coefficients,
        lag_column_budgets=lag_column_budgets,
        contemp_column_budgets=contemp_column_budgets,
        guardc_scale_factors=guardc_scale_factors,
        lag_spectral_radii=lag_spectral_radii,
    )


def sample_signed_budgeted_coefficients(
    edge_mask: np.ndarray,
    target_budgets: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample signed coefficients whose L1 mass per target is budget-constrained.

    Args:
        edge_mask: Boolean edge indicator array with shape (B, p, p) for B blocks
            (e.g., lag blocks) and source->target orientation.
        target_budgets: Array of shape (p,) with non-negative budgets.
        rng: Random generator.
    Returns:
        Coefficients with same shape as edge_mask and support constrained to edge_mask.
    """
    if edge_mask.ndim != 3:
        raise ValueError("edge_mask must have shape (B, p, p).")
    num_blocks, num_sources, num_targets = edge_mask.shape
    if num_sources != num_targets:
        raise ValueError("edge_mask must be square in source/target dimensions.")
    if target_budgets.shape != (num_targets,):
        raise ValueError(
            "target_budgets must have shape "
            f"({num_targets},), got {target_budgets.shape}."
        )
    if np.any(target_budgets < 0):
        raise ValueError("target_budgets must be non-negative.")

    coefficients = np.zeros(edge_mask.shape, dtype=np.float64)
    for target in range(num_targets):
        budget = float(target_budgets[target])
        if budget <= 0.0:
            continue
        active_indices = np.argwhere(edge_mask[:, :, target])
        num_active = active_indices.shape[0]
        if num_active == 0:
            continue

        raw_weights = rng.gamma(shape=1.0, scale=1.0, size=num_active).astype(
            np.float64
        )
        total = float(raw_weights.sum())
        if total <= 0.0:
            continue
        magnitudes = (raw_weights / total) * budget
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=num_active)
        values = magnitudes * signs

        for idx, (block_idx, source_idx) in enumerate(active_indices):
            coefficients[block_idx, source_idx, target] = values[idx]

    return coefficients


def project_to_column_budgets(
    lag_coefficients: np.ndarray,
    target_budgets: np.ndarray,
) -> np.ndarray:
    """Project lag coefficients to per-target L1 column budgets."""
    if lag_coefficients.ndim != 3:
        raise ValueError("lag_coefficients must have shape (L, p, p).")
    _, _, num_targets = lag_coefficients.shape
    if target_budgets.shape != (num_targets,):
        raise ValueError(
            "target_budgets must have shape "
            f"({num_targets},), got {target_budgets.shape}."
        )

    projected = lag_coefficients.astype(np.float64, copy=True)
    l1_per_target = np.sum(np.abs(projected), axis=(0, 1))
    for target in range(num_targets):
        budget = float(target_budgets[target])
        if budget < 0.0:
            raise ValueError("target budgets must be non-negative.")
        norm = float(l1_per_target[target])
        if norm <= budget or norm == 0.0:
            continue
        projected[:, :, target] *= budget / norm
    return projected


def build_companion_matrix(lag_coefficients: np.ndarray) -> np.ndarray:
    """Build companion matrix from lag coefficients.

    The expected orientation is source->target, i.e. coefficient[source, target].
    This is converted internally to the standard row-target form.
    """
    if lag_coefficients.ndim != 3:
        raise ValueError("lag_coefficients must have shape (L, p, p).")
    max_lag, num_sources, num_targets = lag_coefficients.shape
    if num_sources != num_targets:
        raise ValueError("lag_coefficients must have square source/target dimensions.")
    num_vars = num_targets

    companion = np.zeros((num_vars * max_lag, num_vars * max_lag), dtype=np.float64)
    for lag_idx in range(max_lag):
        row_target_block = lag_coefficients[lag_idx].T
        left = lag_idx * num_vars
        right = left + num_vars
        companion[:num_vars, left:right] = row_target_block

    if max_lag > 1:
        companion[num_vars:, :-num_vars] = np.eye(
            num_vars * (max_lag - 1), dtype=np.float64
        )

    return companion


def companion_spectral_radius(lag_coefficients: np.ndarray) -> float:
    """Return the spectral radius of the lag companion matrix."""
    companion = build_companion_matrix(lag_coefficients)
    eigenvalues = np.linalg.eigvals(companion)
    return float(np.max(np.abs(eigenvalues)))


def guardc_rescale_lag_block(
    lag_coefficients: np.ndarray,
    *,
    delta: float,
    max_iter: int = 64,
    step_ceiling: float = 0.98,
) -> tuple[np.ndarray, float, float]:
    """Rescale lag block so companion spectral radius is <= delta."""
    if lag_coefficients.ndim != 3:
        raise ValueError("lag_coefficients must have shape (L, p, p).")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1).")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1.")
    if not 0.0 < step_ceiling < 1.0:
        raise ValueError("step_ceiling must be in (0, 1).")

    scaled = lag_coefficients.astype(np.float64, copy=True)
    rho = companion_spectral_radius(scaled)
    if rho <= delta or rho == 0.0:
        return scaled, 1.0, rho

    factor = 1.0
    for _ in range(max_iter):
        step = min(step_ceiling, delta / (rho + 1e-12))
        factor *= step
        scaled = lag_coefficients * factor
        rho = companion_spectral_radius(scaled)
        if rho <= delta:
            return scaled, factor, rho

    # Final guarded step if numerical noise prevented convergence in the loop.
    factor *= min(step_ceiling, delta / (rho + 1e-12))
    scaled = lag_coefficients * factor
    rho = companion_spectral_radius(scaled)
    return scaled, factor, rho


def project_after_drift(
    cfg: DynSCMConfig,
    lag_coefficients: np.ndarray,
    target_budgets: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Project drifted lag coefficients back into the stable set."""
    projected = project_to_column_budgets(lag_coefficients, target_budgets)
    if cfg.enable_guardc_rescale:
        return guardc_rescale_lag_block(projected, delta=cfg.guardc_delta)
    return projected, 1.0, companion_spectral_radius(projected)


def _validate_stability_sample(
    *,
    cfg: DynSCMConfig,
    graph_sample: DynSCMGraphSample,
    lag_coefficients: np.ndarray,
    contemporaneous_coefficients: np.ndarray,
    lag_column_budgets: np.ndarray,
    contemp_column_budgets: np.ndarray,
    lag_spectral_radii: np.ndarray,
) -> None:
    tol = 1e-8

    expected_lag_shape = graph_sample.regime_lagged_adjacency.shape
    expected_cont_shape = graph_sample.regime_contemporaneous_adjacency.shape
    if lag_coefficients.shape != expected_lag_shape:
        raise RuntimeError(
            f"lag_coefficients shape {lag_coefficients.shape} != {expected_lag_shape}"
        )
    if contemporaneous_coefficients.shape != expected_cont_shape:
        raise RuntimeError(
            "contemporaneous_coefficients shape "
            f"{contemporaneous_coefficients.shape} != {expected_cont_shape}"
        )

    off_support_lag = (~graph_sample.regime_lagged_adjacency) & (
        np.abs(lag_coefficients) > tol
    )
    off_support_cont = (~graph_sample.regime_contemporaneous_adjacency) & (
        np.abs(contemporaneous_coefficients) > tol
    )
    if off_support_lag.any():
        raise RuntimeError(
            "Lag coefficients contain non-zero values outside graph support."
        )
    if off_support_cont.any():
        raise RuntimeError(
            "Contemporaneous coefficients contain non-zero values outside "
            "graph support."
        )

    lag_l1 = np.sum(np.abs(lag_coefficients), axis=(1, 2))
    cont_l1 = np.sum(np.abs(contemporaneous_coefficients), axis=1)
    if np.any(lag_l1 > lag_column_budgets + tol):
        raise RuntimeError("Lag L1 budgets violated.")
    if np.any(cont_l1 > contemp_column_budgets + tol):
        raise RuntimeError("Contemporaneous L1 budgets violated.")

    if np.any(lag_column_budgets < cfg.col_budget_min - tol):
        raise RuntimeError("Sampled lag budgets below minimum.")
    if np.any(lag_column_budgets > cfg.col_budget_max + tol):
        raise RuntimeError("Sampled lag budgets above maximum.")
    if np.any(contemp_column_budgets < -tol):
        raise RuntimeError("Sampled contemporaneous budgets below zero.")
    if np.any(contemp_column_budgets > cfg.contemp_budget_max + tol):
        raise RuntimeError("Sampled contemporaneous budgets above maximum.")

    if cfg.enable_guardc_rescale and np.any(
        lag_spectral_radii > cfg.guardc_delta + 1e-6
    ):
        raise RuntimeError("Guard C spectral radius constraint violated.")
