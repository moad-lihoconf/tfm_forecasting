"""Simulation engine for DynSCM time-series generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DynSCMConfig
from .graph import DynSCMGraphSample
from .mechanisms import DynSCMMechanismSample, evaluate_lagged_mechanism


@dataclass(frozen=True, slots=True)
class DynSCMSimulationSample:
    """Forward simulation output for one multivariate DynSCM trajectory."""

    series: np.ndarray  # (T, p), simulated observations.
    regime_path: np.ndarray  # (T,), regime index at each time.
    innovations: np.ndarray  # (T, p), sampled additive innovations.
    noise_scales: np.ndarray  # (p,), per-variable innovation scales.
    initial_lagged_history: np.ndarray  # (L, p), row 0 is lag-1 history.
    clipped: bool  # Whether final clipping fallback was applied.
    num_attempts: int  # Number of attempts consumed before returning.
    max_abs_value: float  # max(abs(series)) in the returned sample.

    @property
    def num_steps(self) -> int:
        return self.series.shape[0]

    @property
    def num_vars(self) -> int:
        return self.series.shape[1]


def sample_regime_path(
    cfg: DynSCMConfig,
    num_steps: int,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample a sticky Markov regime path with values in [0, K)."""

    generator = cfg.make_rng(seed) if rng is None else rng
    num_regimes = int(cfg.num_regimes)

    path = np.empty((num_steps,), dtype=np.int64)
    if num_regimes == 1:
        path.fill(0)
        return path
    path[0] = int(generator.integers(0, num_regimes))

    for step_idx in range(1, num_steps):
        previous = int(path[step_idx - 1])
        if generator.random() < cfg.sticky_rho:
            path[step_idx] = previous
            continue
        sampled = int(generator.integers(0, num_regimes - 1))
        path[step_idx] = sampled + int(sampled >= previous)

    return path


def sample_innovations(
    cfg: DynSCMConfig,
    num_steps: int,
    num_vars: int,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample additive innovations and per-variable scales."""

    generator = cfg.make_rng(seed) if rng is None else rng
    noise_scales = generator.uniform(
        cfg.noise_scale_min,
        cfg.noise_scale_max,
        size=(num_vars,),
    ).astype(np.float64)

    if cfg.noise_family == "normal":
        base = generator.normal(size=(num_steps, num_vars)).astype(np.float64)
    elif cfg.noise_family == "student_t":
        base = generator.standard_t(
            df=cfg.student_df,
            size=(num_steps, num_vars),
        ).astype(np.float64)
        # Standardize Student-t to unit variance before applying noise scales.
        base *= np.sqrt((cfg.student_df - 2.0) / cfg.student_df)
    else:
        raise ValueError(f"Unsupported noise_family={cfg.noise_family!r}.")

    innovations = base * noise_scales[None, :]
    return innovations, noise_scales


def simulate_dynscm_series(
    cfg: DynSCMConfig,
    graph_sample: DynSCMGraphSample,
    mechanism_sample: DynSCMMechanismSample,
    *,
    num_steps: int | None = None,
    regime_path: np.ndarray | None = None,
    seed: int | None = None,
) -> DynSCMSimulationSample:
    """Forward-simulate X_{1:T} with bounded retries and clipping fallback."""
    _validate_graph_mechanism_match(graph_sample, mechanism_sample)

    rng = cfg.make_rng(seed)
    resolved_num_steps = _resolve_num_steps(cfg, num_steps, rng)
    resolved_regime_path = _resolve_regime_path(
        cfg,
        resolved_num_steps,
        regime_path=regime_path,
    )

    max_attempts = int(max(1, cfg.max_resample_attempts))
    fallback: (
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int] | None
    ) = None

    for attempt_idx in range(1, max_attempts + 1):
        current_path = (
            resolved_regime_path.copy()
            if resolved_regime_path is not None
            else sample_regime_path(cfg, resolved_num_steps, rng=rng)
        )
        innovations, noise_scales = sample_innovations(
            cfg,
            resolved_num_steps,
            graph_sample.num_vars,
            rng=rng,
        )
        initial_history = rng.normal(
            loc=0.0,
            scale=float(np.mean(noise_scales)),
            size=(graph_sample.max_lag, graph_sample.num_vars),
        ).astype(np.float64)

        series = _simulate_forward(
            graph_sample,
            mechanism_sample,
            regime_path=current_path,
            innovations=innovations,
            initial_lagged_history=initial_history,
        )

        max_abs = float(np.max(np.abs(series)))
        if np.isfinite(series).all() and max_abs <= cfg.max_abs_x:
            return DynSCMSimulationSample(
                series=series,
                regime_path=current_path,
                innovations=innovations,
                noise_scales=noise_scales,
                initial_lagged_history=initial_history,
                clipped=False,
                num_attempts=attempt_idx,
                max_abs_value=max_abs,
            )

        fallback = (
            series,
            current_path,
            innovations,
            noise_scales,
            initial_history,
            attempt_idx,
        )

    assert fallback is not None

    (
        raw_series,
        path,
        innovations,
        noise_scales,
        initial_history,
        used_attempts,
    ) = fallback
    clipped_series = np.clip(
        np.nan_to_num(
            raw_series,
            nan=0.0,
            posinf=cfg.max_abs_x,
            neginf=-cfg.max_abs_x,
        ),
        -cfg.max_abs_x,
        cfg.max_abs_x,
    ).astype(np.float64)

    return DynSCMSimulationSample(
        series=clipped_series,
        regime_path=path,
        innovations=innovations,
        noise_scales=noise_scales,
        initial_lagged_history=initial_history,
        clipped=True,
        num_attempts=used_attempts,
        max_abs_value=float(np.max(np.abs(clipped_series))),
    )


def _resolve_num_steps(
    cfg: DynSCMConfig,
    num_steps: int | None,
    rng: np.random.Generator,
) -> int:
    if num_steps is not None:
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1.")
        return int(num_steps)
    return int(rng.integers(cfg.series_length_min, cfg.series_length_max + 1))


def _resolve_regime_path(
    cfg: DynSCMConfig,
    num_steps: int,
    *,
    regime_path: np.ndarray | None,
) -> np.ndarray | None:
    if regime_path is None:
        return None
    path = np.asarray(regime_path, dtype=np.int64)
    if path.shape != (num_steps,):
        raise ValueError(f"regime_path shape {path.shape} must be ({num_steps},).")
    if path.min() < 0 or path.max() >= cfg.num_regimes:
        raise ValueError("regime_path values must be in [0, num_regimes).")
    # Copy to ensure caller-owned arrays are never mutated internally.
    return path.copy()


def _simulate_forward(
    graph_sample: DynSCMGraphSample,
    mechanism_sample: DynSCMMechanismSample,
    *,
    regime_path: np.ndarray,
    innovations: np.ndarray,
    initial_lagged_history: np.ndarray,
) -> np.ndarray:
    num_steps = int(innovations.shape[0])
    num_vars = int(innovations.shape[1])
    max_lag = int(graph_sample.max_lag)

    if regime_path.shape != (num_steps,):
        raise ValueError(
            f"regime_path shape {regime_path.shape} must be ({num_steps},)."
        )
    if initial_lagged_history.shape != (max_lag, num_vars):
        raise ValueError(
            "initial_lagged_history shape "
            f"{initial_lagged_history.shape} must be ({max_lag}, {num_vars})."
        )

    lagged_history = initial_lagged_history.astype(np.float64, copy=True)
    series = np.zeros((num_steps, num_vars), dtype=np.float64)

    for step_idx in range(num_steps):
        regime_idx = int(regime_path[step_idx])

        lagged_response = evaluate_lagged_mechanism(
            mechanism_sample,
            regime_idx=regime_idx,
            lagged_history=lagged_history,
        )
        base_values = lagged_response + innovations[step_idx]
        series[step_idx] = _apply_contemporaneous_step(
            base_values=base_values,
            contemp_coeffs=mechanism_sample.contemp_coeffs[regime_idx],
            topo_order=graph_sample.regime_topo_orders[regime_idx],
        )

        lagged_history[1:] = lagged_history[:-1]
        lagged_history[0] = series[step_idx]

    return series


def _apply_contemporaneous_step(
    *,
    base_values: np.ndarray,
    contemp_coeffs: np.ndarray,
    topo_order: np.ndarray,
) -> np.ndarray:
    current = np.zeros_like(base_values, dtype=np.float64)
    for target_raw in topo_order:
        target = int(target_raw)
        parent_coeffs = contemp_coeffs[:, target]
        parent_mask = np.abs(parent_coeffs) > 0.0
        if parent_mask.any():
            current[target] = base_values[target] + float(
                np.dot(current[parent_mask], parent_coeffs[parent_mask])
            )
        else:
            current[target] = base_values[target]
    return current


def _validate_graph_mechanism_match(
    graph_sample: DynSCMGraphSample,
    mechanism_sample: DynSCMMechanismSample,
) -> None:
    if graph_sample.num_regimes != mechanism_sample.num_regimes:
        raise ValueError("num_regimes mismatch between graph and mechanism samples.")
    if graph_sample.num_vars != mechanism_sample.num_vars:
        raise ValueError("num_vars mismatch between graph and mechanism samples.")
    if graph_sample.max_lag != mechanism_sample.max_lag:
        raise ValueError("max_lag mismatch between graph and mechanism samples.")
