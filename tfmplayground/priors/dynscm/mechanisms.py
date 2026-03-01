"""Mechanism sampling utilities for DynSCM."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DynSCMConfig
from .graph import DynSCMGraphSample
from .stability import DynSCMStabilitySample, sample_stable_coefficients


@dataclass(frozen=True, slots=True)
class DynSCMMechanismSample:
    """Sampled regime mechanisms: stable linear core + optional residual block."""

    mechanism_type: str
    lag_coeffs: np.ndarray  # (K, L, p, p), source->target orientation.
    contemp_coeffs: np.ndarray  # (K, p, p), source->target orientation.
    residual_input_weights: np.ndarray  # (K, p, M, L*p), random-feature projections.
    residual_biases: np.ndarray  # (K, p, M), random-feature biases.
    residual_output_weights: np.ndarray  # (K, p, M), feature combination weights.
    residual_lipschitz_upper_bounds: (
        np.ndarray
    )  # (K, p), per-node analytic upper bounds.

    @property
    def num_regimes(self) -> int:
        return self.lag_coeffs.shape[0]

    @property
    def max_lag(self) -> int:
        return self.lag_coeffs.shape[1]

    @property
    def num_vars(self) -> int:
        return self.lag_coeffs.shape[2]

    @property
    def residual_num_features(self) -> int:
        return int(self.residual_output_weights.shape[-1])

    @property
    def has_residual(self) -> bool:
        return self.residual_num_features > 0


def sample_regime_mechanisms(
    cfg: DynSCMConfig,
    graph_sample: DynSCMGraphSample,
    *,
    stability_sample: DynSCMStabilitySample | None = None,
    target_idx: int | None = None,
    seed: int | None = None,
) -> DynSCMMechanismSample:
    """Sample phase-4 mechanisms coupled to graph and stability outputs.

    The linear lag/contemporaneous core is sourced from `stability_sample`.
    If absent, a fresh stable sample is drawn deterministically from the given seed.
    """
    if cfg.mechanism_type not in ("linear_var", "linear_plus_residual"):
        raise ValueError(f"Unsupported mechanism_type={cfg.mechanism_type!r}.")

    if stability_sample is None:
        rng = cfg.make_rng(seed)
        stability_seed = int(rng.integers(0, np.iinfo(np.int64).max))
        stability_sample = sample_stable_coefficients(
            cfg, graph_sample, target_idx=target_idx, seed=stability_seed
        )
    else:
        rng = cfg.make_rng(seed)

    _validate_graph_stability_match(cfg, graph_sample, stability_sample)

    (
        residual_input_weights,
        residual_biases,
        residual_output_weights,
        lipschitz_bounds,
    ) = _sample_residual_parameters(
        cfg,
        num_regimes=graph_sample.num_regimes,
        num_vars=graph_sample.num_vars,
        max_lag=graph_sample.max_lag,
        rng=rng,
    )

    mechanism_sample = DynSCMMechanismSample(
        mechanism_type=cfg.mechanism_type,
        lag_coeffs=stability_sample.lag_coeffs.copy(),
        contemp_coeffs=stability_sample.contemp_coeffs.copy(),
        residual_input_weights=residual_input_weights,
        residual_biases=residual_biases,
        residual_output_weights=residual_output_weights,
        residual_lipschitz_upper_bounds=lipschitz_bounds,
    )
    _validate_mechanism_sample(cfg, mechanism_sample)
    return mechanism_sample


def evaluate_lagged_mechanism(
    mechanism_sample: DynSCMMechanismSample,
    *,
    regime_idx: int,
    lagged_history: np.ndarray,
) -> np.ndarray:
    """Evaluate lagged mechanism response for one regime and one time step.

    Args:
        mechanism_sample: Mechanism container sampled via `sample_regime_mechanisms`.
        regime_idx: Regime index in [0, K).
        lagged_history: Array of shape (L, p), where row 0 is lag-1, row 1 lag-2, etc.
    Returns:
        Vector of shape (p,) with lagged contributions for each target node.
    """
    if not 0 <= regime_idx < mechanism_sample.num_regimes:
        raise ValueError(
            f"regime_idx={regime_idx} outside [0, {mechanism_sample.num_regimes})."
        )
    expected = (mechanism_sample.max_lag, mechanism_sample.num_vars)
    if lagged_history.shape != expected:
        raise ValueError(
            f"lagged_history shape {lagged_history.shape} must be {expected}."
        )

    lag_block = mechanism_sample.lag_coeffs[regime_idx]
    linear_response = np.einsum("ls,lst->t", lagged_history, lag_block, optimize=True)

    if not mechanism_sample.has_residual:
        return linear_response

    flat_history = lagged_history.reshape(-1)
    input_weights = mechanism_sample.residual_input_weights[regime_idx]
    biases = mechanism_sample.residual_biases[regime_idx]
    output_weights = mechanism_sample.residual_output_weights[regime_idx]

    hidden = np.tanh(
        np.einsum("jmh,h->jm", input_weights, flat_history, optimize=True) + biases
    )
    residual_response = np.sum(output_weights * hidden, axis=1)
    return linear_response + residual_response


def evaluate_contemporaneous_effect(
    mechanism_sample: DynSCMMechanismSample,
    *,
    regime_idx: int,
    current_state: np.ndarray,
) -> np.ndarray:
    """Apply contemporaneous source->target linear map for one regime."""
    if not 0 <= regime_idx < mechanism_sample.num_regimes:
        raise ValueError(
            f"regime_idx={regime_idx} outside [0, {mechanism_sample.num_regimes})."
        )
    expected = (mechanism_sample.num_vars,)
    if current_state.shape != expected:
        raise ValueError(
            f"current_state shape {current_state.shape} must be {expected}."
        )
    return current_state @ mechanism_sample.contemp_coeffs[regime_idx]


def _sample_residual_parameters(
    cfg: DynSCMConfig,
    *,
    num_regimes: int,
    num_vars: int,
    max_lag: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_features = (
        cfg.residual_num_features if cfg.mechanism_type == "linear_plus_residual" else 0
    )
    feature_input_dim = max_lag * num_vars
    if num_features == 0:
        empty_inputs = np.zeros(
            (num_regimes, num_vars, 0, feature_input_dim), dtype=np.float64
        )
        empty_biases = np.zeros((num_regimes, num_vars, 0), dtype=np.float64)
        empty_outputs = np.zeros((num_regimes, num_vars, 0), dtype=np.float64)
        zero_bounds = np.zeros((num_regimes, num_vars), dtype=np.float64)
        return empty_inputs, empty_biases, empty_outputs, zero_bounds

    input_weights = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(max(feature_input_dim, 1)),
        size=(num_regimes, num_vars, num_features, feature_input_dim),
    ).astype(np.float64)
    biases = rng.normal(
        loc=0.0,
        scale=0.25,
        size=(num_regimes, num_vars, num_features),
    ).astype(np.float64)
    output_weights = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(max(num_features, 1)),
        size=(num_regimes, num_vars, num_features),
    ).astype(np.float64)

    lipschitz_bounds = _residual_lipschitz_upper_bounds(input_weights, output_weights)
    cap = cfg.residual_lipschitz_max
    if cap <= 0.0:
        output_weights.fill(0.0)
    else:
        scales = np.minimum(1.0, cap / (lipschitz_bounds + 1e-12))
        output_weights *= scales[..., None]
    lipschitz_bounds = _residual_lipschitz_upper_bounds(input_weights, output_weights)
    return input_weights, biases, output_weights, lipschitz_bounds


def _residual_lipschitz_upper_bounds(
    input_weights: np.ndarray,
    output_weights: np.ndarray,
) -> np.ndarray:
    """Compute analytic L-infinity Lipschitz upper bounds per node.

    For each node j: sum_m |a_{j,m}| * ||w_{j,m}||_1.
    """
    if input_weights.ndim != 4:
        raise ValueError("input_weights must have shape (K, p, M, D).")
    if output_weights.ndim != 3:
        raise ValueError("output_weights must have shape (K, p, M).")
    if input_weights.shape[:3] != output_weights.shape:
        raise ValueError(
            "input_weights and output_weights disagree on (K, p, M) dimensions."
        )

    input_l1 = np.sum(np.abs(input_weights), axis=-1)
    return np.sum(np.abs(output_weights) * input_l1, axis=-1)


def _validate_graph_stability_match(
    cfg: DynSCMConfig,
    graph_sample: DynSCMGraphSample,
    stability_sample: DynSCMStabilitySample,
) -> None:
    if stability_sample.lag_coeffs.shape != graph_sample.regime_lagged_adjacency.shape:
        raise ValueError(
            "stability lag_coeffs shape does not match graph lagged adjacency."
        )
    if (
        stability_sample.contemp_coeffs.shape
        != graph_sample.regime_contemp_adjacency.shape
    ):
        raise ValueError(
            "stability contemporaneous_coefficients shape does not match graph "
            "contemporaneous adjacency."
        )


def _validate_mechanism_sample(
    cfg: DynSCMConfig,
    mechanism_sample: DynSCMMechanismSample,
) -> None:
    tol = 1e-8
    k = mechanism_sample.num_regimes
    p = mechanism_sample.num_vars
    max_lag = mechanism_sample.max_lag
    m = mechanism_sample.residual_num_features
    d = max_lag * p

    if mechanism_sample.lag_coeffs.shape != (k, max_lag, p, p):
        raise RuntimeError("Invalid lag_coeffs shape.")
    if mechanism_sample.contemp_coeffs.shape != (k, p, p):
        raise RuntimeError("Invalid contemporaneous_coefficients shape.")
    if mechanism_sample.residual_input_weights.shape != (k, p, m, d):
        raise RuntimeError("Invalid residual_input_weights shape.")
    if mechanism_sample.residual_biases.shape != (k, p, m):
        raise RuntimeError("Invalid residual_biases shape.")
    if mechanism_sample.residual_output_weights.shape != (k, p, m):
        raise RuntimeError("Invalid residual_output_weights shape.")
    if mechanism_sample.residual_lipschitz_upper_bounds.shape != (k, p):
        raise RuntimeError("Invalid residual_lipschitz_upper_bounds shape.")

    arrays = (
        mechanism_sample.lag_coeffs,
        mechanism_sample.contemp_coeffs,
        mechanism_sample.residual_input_weights,
        mechanism_sample.residual_biases,
        mechanism_sample.residual_output_weights,
        mechanism_sample.residual_lipschitz_upper_bounds,
    )
    if any(not np.isfinite(arr).all() for arr in arrays):
        raise RuntimeError("Mechanism arrays must be finite.")

    if mechanism_sample.mechanism_type == "linear_var" and m != 0:
        raise RuntimeError("linear_var mode must not include residual features.")
    if mechanism_sample.mechanism_type == "linear_plus_residual":
        cap = cfg.residual_lipschitz_max
        if np.any(mechanism_sample.residual_lipschitz_upper_bounds > cap + tol):
            raise RuntimeError("Residual Lipschitz bounds exceed configured cap.")
