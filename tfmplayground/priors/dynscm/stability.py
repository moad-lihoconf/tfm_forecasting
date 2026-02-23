"""Stability utilities for DynSCM coefficient sampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DynSCMConfig
from .graph import DynSCMGraphSample


@dataclass(frozen=True, slots=True)
class DynSCMStabilitySample:
    """Sampled stable coefficients coupled to a DynSCM regime graph."""

    lag_coeffs: np.ndarray  # (K, L, p, p), source->target coefficients.
    contemp_coeffs: np.ndarray  # (K, p, p), source->target coefficients.
    lag_column_budgets: np.ndarray  # (K, p), per-target budgets for lag edges.
    contemp_column_budgets: np.ndarray  # (K, p), per-target budgets for cont. edges.
    spectral_rescale_factors: np.ndarray  # (K,), multiplicative lag rescale factors.
    lag_spectral_radii: np.ndarray  # (K,), spectral radius after optional rescaling.

    @property
    def num_regimes(self) -> int:
        return self.lag_coeffs.shape[0]

    @property
    def max_lag(self) -> int:
        return self.lag_coeffs.shape[1]

    @property
    def num_vars(self) -> int:
        return self.lag_coeffs.shape[2]


def sample_stable_coefficients(
    cfg: DynSCMConfig,
    graph_sample: DynSCMGraphSample,
    *,
    seed: int | None = None,
) -> DynSCMStabilitySample:
    num_regimes = graph_sample.num_regimes
    max_lag = graph_sample.max_lag
    num_vars = graph_sample.num_vars

    rng = cfg.make_rng(seed)

    lag_coeffs = np.zeros((num_regimes, max_lag, num_vars, num_vars), dtype=np.float64)
    contemp_coeffs = np.zeros((num_regimes, num_vars, num_vars), dtype=np.float64)
    lag_column_budgets = np.zeros((num_regimes, num_vars), dtype=np.float64)
    contemp_column_budgets = np.zeros((num_regimes, num_vars), dtype=np.float64)
    spectral_rescale_factors = np.ones((num_regimes,), dtype=np.float64)
    lag_spectral_radii = np.zeros((num_regimes,), dtype=np.float64)

    for regime_idx in range(num_regimes):
        lag_budget = rng.uniform(
            cfg.col_budget_min, cfg.col_budget_max, size=(num_vars,)
        ).astype(np.float64)
        cont_budget = rng.uniform(0.0, cfg.contemp_budget_max, size=(num_vars,)).astype(
            np.float64
        )

        lag_mask = graph_sample.regime_lagged_adjacency[regime_idx]
        cont_mask = graph_sample.regime_contemp_adjacency[regime_idx]

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

        if cfg.enable_spectral_rescale:
            lag_block, scale, rho = rescale_lag_block_to_spectral_radius(
                lag_coeffs=lag_block,
                spectral_radius_cap=cfg.spectral_radius_cap,
            )
            spectral_rescale_factors[regime_idx] = scale
            lag_spectral_radii[regime_idx] = rho
        else:
            lag_spectral_radii[regime_idx] = companion_spectral_radius(lag_block)

        lag_coeffs[regime_idx] = lag_block
        contemp_coeffs[regime_idx] = cont_block
        lag_column_budgets[regime_idx] = lag_budget
        contemp_column_budgets[regime_idx] = cont_budget

    _validate_stability_sample(
        cfg=cfg,
        graph_sample=graph_sample,
        lag_coeffs=lag_coeffs,
        contemp_coeffs=contemp_coeffs,
        lag_column_budgets=lag_column_budgets,
        contemp_column_budgets=contemp_column_budgets,
        lag_spectral_radii=lag_spectral_radii,
    )

    return DynSCMStabilitySample(
        lag_coeffs=lag_coeffs,
        contemp_coeffs=contemp_coeffs,
        lag_column_budgets=lag_column_budgets,
        contemp_column_budgets=contemp_column_budgets,
        spectral_rescale_factors=spectral_rescale_factors,
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

    _, _, num_targets = edge_mask.shape

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
    lag_coeffs: np.ndarray,
    target_budgets: np.ndarray,
) -> np.ndarray:
    """Project lag coefficients to per-target L1 column budgets."""

    _, _, num_targets = lag_coeffs.shape
    if target_budgets.shape != (num_targets,):
        raise ValueError(
            "target_budgets must have shape "
            f"({num_targets},), got {target_budgets.shape}."
        )

    projected = lag_coeffs.astype(np.float64, copy=True)
    l1_per_target = np.sum(np.abs(projected), axis=(0, 1))
    for target in range(num_targets):
        budget = float(target_budgets[target])
        norm = float(l1_per_target[target])
        if norm <= budget or norm == 0.0:
            continue
        projected[:, :, target] *= budget / norm
    return projected


def build_companion_matrix(lag_coeffs: np.ndarray) -> np.ndarray:
    """Build companion matrix from lag coefficients.

    The expected orientation is source->target, i.e. coefficient[source, target].
    This is converted internally to the standard row-target form.
    """
    max_lag, num_sources, num_targets = lag_coeffs.shape
    if num_sources != num_targets:
        raise ValueError("lag_coefficients must have square source/target dimensions.")
    num_vars = num_targets

    companion = np.zeros((num_vars * max_lag, num_vars * max_lag), dtype=np.float64)
    for lag_idx in range(max_lag):
        row_target_block = lag_coeffs[lag_idx].T
        left = lag_idx * num_vars
        right = left + num_vars
        companion[:num_vars, left:right] = row_target_block

    if max_lag > 1:
        companion[num_vars:, :-num_vars] = np.eye(
            num_vars * (max_lag - 1), dtype=np.float64
        )

    return companion


def companion_spectral_radius(lag_coeffs: np.ndarray) -> float:
    """Return the spectral radius of the lag companion matrix."""
    companion = build_companion_matrix(lag_coeffs)
    eigenvalues = np.linalg.eigvals(companion)
    return float(np.max(np.abs(eigenvalues)))


def rescale_lag_block_to_spectral_radius(
    lag_coeffs: np.ndarray,
    *,
    spectral_radius_cap: float,
    max_iter: int = 64,
    step_ceiling: float = 0.98,
) -> tuple[np.ndarray, float, float]:
    """Rescale lag block so companion spectral radius is <= spectral_radius_cap."""

    scaled = lag_coeffs.astype(np.float64, copy=True)
    rho = companion_spectral_radius(scaled)
    if rho <= spectral_radius_cap or rho == 0.0:
        return scaled, 1.0, rho

    factor = 1.0
    for _ in range(max_iter):
        step = min(step_ceiling, spectral_radius_cap / (rho + 1e-12))
        factor *= step
        scaled = lag_coeffs * factor
        rho = companion_spectral_radius(scaled)
        if rho <= spectral_radius_cap:
            return scaled, factor, rho

    # Final fallback step if numerical noise prevented convergence in the loop.
    factor *= min(step_ceiling, spectral_radius_cap / (rho + 1e-12))
    scaled = lag_coeffs * factor
    rho = companion_spectral_radius(scaled)
    return scaled, factor, rho


def project_after_drift(
    cfg: DynSCMConfig,
    lag_coeffs: np.ndarray,
    target_budgets: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Project drifted lag coefficients back into the stable set."""
    projected = project_to_column_budgets(lag_coeffs, target_budgets)
    if cfg.enable_spectral_rescale:
        return rescale_lag_block_to_spectral_radius(
            projected, spectral_radius_cap=cfg.spectral_radius_cap
        )
    return projected, 1.0, companion_spectral_radius(projected)


def _validate_stability_sample(
    *,
    cfg: DynSCMConfig,
    graph_sample: DynSCMGraphSample,
    lag_coeffs: np.ndarray,
    contemp_coeffs: np.ndarray,
    lag_column_budgets: np.ndarray,
    contemp_column_budgets: np.ndarray,
    lag_spectral_radii: np.ndarray,
) -> None:
    tol = 1e-8

    expected_lag_shape = graph_sample.regime_lagged_adjacency.shape
    expected_cont_shape = graph_sample.regime_contemp_adjacency.shape
    if lag_coeffs.shape != expected_lag_shape:
        raise RuntimeError(
            f"lag_coefficients shape {lag_coeffs.shape} != {expected_lag_shape}"
        )
    if contemp_coeffs.shape != expected_cont_shape:
        raise RuntimeError(
            "contemporaneous_coefficients shape "
            f"{contemp_coeffs.shape} != {expected_cont_shape}"
        )

    off_support_lag = (~graph_sample.regime_lagged_adjacency) & (
        np.abs(lag_coeffs) > tol
    )
    off_support_cont = (~graph_sample.regime_contemp_adjacency) & (
        np.abs(contemp_coeffs) > tol
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

    lag_l1 = np.sum(np.abs(lag_coeffs), axis=(1, 2))
    cont_l1 = np.sum(np.abs(contemp_coeffs), axis=1)
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

    if cfg.enable_spectral_rescale and np.any(
        lag_spectral_radii > cfg.spectral_radius_cap + 1e-6
    ):
        raise RuntimeError("Spectral radius constraint violated.")
