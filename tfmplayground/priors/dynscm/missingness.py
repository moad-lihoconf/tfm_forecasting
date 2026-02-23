"""Raw-series observation mask sampling for DynSCM."""

from __future__ import annotations

import numpy as np

from .config import DynSCMConfig

# MAR and MNAR logit weights — used by both standalone modes and the mix sampler
# to guarantee consistency when weights are updated.
_MAR_TIME_WEIGHT = 0.8
_MAR_SUMMARY_WEIGHT = 0.9
_MNAR_MAG_WEIGHT = 1.0


def _mar_step_logits(
    base_logits: np.ndarray,
    time_norm: float,
    observed_abs_sum: np.ndarray,
    observed_count: np.ndarray,
) -> np.ndarray:
    """Return MAR missing logits for one time step given the running obs summary."""
    observed_mean_abs = observed_abs_sum / np.maximum(observed_count, 1.0)
    return (
        base_logits
        + _MAR_TIME_WEIGHT * (2.0 * time_norm - 1.0)
        + _MAR_SUMMARY_WEIGHT * np.tanh(observed_mean_abs)
    )


def _mnar_step_logits(
    base_logits: np.ndarray,
    current_abs: np.ndarray,
    value_scales: np.ndarray,
) -> np.ndarray:
    """Return MNAR missing logits for one time step given current absolute values."""
    magnitude_ratio = current_abs / value_scales
    return base_logits + _MNAR_MAG_WEIGHT * (magnitude_ratio - 1.0)


def sample_observation_mask(
    cfg: DynSCMConfig,
    series: np.ndarray,
    *,
    seed: int | None = None,
    label_times: np.ndarray | None = None,
    label_var_indices: np.ndarray | int | None = None,
) -> np.ndarray:
    """Sample raw observation mask `obs_mask[B, T, p]` with boolean dtype.

    The mask is intended for feature extraction inputs only.
    If `label_times` and `label_var_indices` are provided, those label positions
    are force-marked as observed to prevent accidental label masking.
    """
    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("series must have shape (B, T, p).")
    batch_size, num_steps, num_vars = values.shape
    if batch_size < 1 or num_steps < 1 or num_vars < 1:
        raise ValueError("series shape must satisfy B>=1, T>=1, p>=1.")

    mode = str(cfg.missing_mode)
    rng = cfg.make_rng(seed)

    if mode == "off":
        obs_mask = np.ones((batch_size, num_steps, num_vars), dtype=bool)
    else:
        base_missing_rates = rng.uniform(
            cfg.missing_rate_min,
            cfg.missing_rate_max,
            size=(batch_size, num_vars),
        ).astype(np.float64)

        if mode == "mcar":
            obs_mask = _sample_mcar_mask(base_missing_rates, num_steps, rng)
        elif mode == "mar":
            obs_mask = _sample_mar_mask(values, base_missing_rates, rng)
        elif mode == "mnar_lite":
            obs_mask = _sample_mnar_lite_mask(values, base_missing_rates, rng)
        elif mode == "mix":
            obs_mask = _sample_mix_mask(values, base_missing_rates, rng)
        else:
            raise ValueError(f"Unsupported missing_mode={mode!r}.")

        _apply_block_outages(
            obs_mask,
            block_missing_prob=cfg.block_missing_prob,
            block_len_min=cfg.block_len_min,
            block_len_max=cfg.block_len_max,
            rng=rng,
        )

    _enforce_label_visibility(
        obs_mask,
        label_times=label_times,
        label_var_indices=label_var_indices,
    )
    return obs_mask.astype(bool, copy=False)


def _sample_mcar_mask(
    base_missing_rates: np.ndarray,
    num_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    batch_size, num_vars = base_missing_rates.shape
    missing_draws = rng.random((batch_size, num_steps, num_vars))
    return missing_draws >= base_missing_rates[:, None, :]


def _sample_mar_mask(
    values: np.ndarray,
    base_missing_rates: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    batch_size, num_steps, num_vars = values.shape
    base_logits = _logit(base_missing_rates)

    obs_mask = np.ones((batch_size, num_steps, num_vars), dtype=bool)
    observed_abs_sum = np.zeros((batch_size, num_vars), dtype=np.float64)
    observed_count = np.zeros((batch_size, num_vars), dtype=np.float64)

    safe_values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    for step_idx in range(num_steps):
        time_norm = step_idx / (num_steps - 1) if num_steps > 1 else 0.0
        logits = _mar_step_logits(
            base_logits, time_norm, observed_abs_sum, observed_count
        )
        missing_prob = _sigmoid(logits)
        observed = rng.random((batch_size, num_vars)) >= missing_prob
        obs_mask[:, step_idx, :] = observed

        current_abs = np.abs(safe_values[:, step_idx, :])
        observed_abs_sum += current_abs * observed
        observed_count += observed

    return obs_mask


def _sample_mnar_lite_mask(
    values: np.ndarray,
    base_missing_rates: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    batch_size, num_steps, num_vars = values.shape
    base_logits = _logit(base_missing_rates)

    safe_values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    value_scales = np.std(safe_values, axis=1) + 1e-6
    magnitude_ratio = np.abs(safe_values) / value_scales[:, None, :]
    logits = base_logits[:, None, :] + _MNAR_MAG_WEIGHT * (magnitude_ratio - 1.0)
    missing_prob = _sigmoid(logits)

    missing_draws = rng.random((batch_size, num_steps, num_vars))
    return missing_draws >= missing_prob


def _sample_mix_mask(
    values: np.ndarray,
    base_missing_rates: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample convex mixture of MCAR/MAR/MNAR-lite missingness."""
    batch_size, num_steps, num_vars = values.shape
    base_logits = _logit(base_missing_rates)
    mix_weights = rng.dirichlet(
        np.array([1.0, 1.0, 1.0], dtype=np.float64),
        size=batch_size,
    )

    safe_values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    value_scales = np.std(safe_values, axis=1) + 1e-6
    # MCAR probability is the base rate directly (no time or magnitude dependence).
    p_mcar = base_missing_rates

    obs_mask = np.ones((batch_size, num_steps, num_vars), dtype=bool)
    observed_abs_sum = np.zeros((batch_size, num_vars), dtype=np.float64)
    observed_count = np.zeros((batch_size, num_vars), dtype=np.float64)

    for step_idx in range(num_steps):
        time_norm = step_idx / (num_steps - 1) if num_steps > 1 else 0.0
        current_abs = np.abs(safe_values[:, step_idx, :])

        p_mar = _sigmoid(
            _mar_step_logits(base_logits, time_norm, observed_abs_sum, observed_count)
        )
        p_mnar = _sigmoid(_mnar_step_logits(base_logits, current_abs, value_scales))

        missing_prob = (
            mix_weights[:, 0, None] * p_mcar
            + mix_weights[:, 1, None] * p_mar
            + mix_weights[:, 2, None] * p_mnar
        )

        observed = rng.random((batch_size, num_vars)) >= missing_prob
        obs_mask[:, step_idx, :] = observed

        observed_abs_sum += current_abs * observed
        observed_count += observed

    return obs_mask


def _apply_block_outages(
    obs_mask: np.ndarray,
    *,
    block_missing_prob: float,
    block_len_min: int,
    block_len_max: int,
    rng: np.random.Generator,
) -> None:
    if block_missing_prob <= 0.0:
        return

    batch_size, num_steps, num_vars = obs_mask.shape
    for batch_idx in range(batch_size):
        for var_idx in range(num_vars):
            if rng.random() >= block_missing_prob:
                continue

            block_len = int(rng.integers(block_len_min, block_len_max + 1))
            block_len = min(block_len, num_steps)
            start_max = num_steps - block_len
            start = int(rng.integers(0, start_max + 1))
            obs_mask[batch_idx, start : start + block_len, var_idx] = False


def _enforce_label_visibility(
    obs_mask: np.ndarray,
    *,
    label_times: np.ndarray | None,
    label_var_indices: np.ndarray | int | None,
) -> None:
    """Force label positions to observed=True.

    Constraints:
    - label_times: shape (B, N_labels), rectangular — pad if counts differ per item.
    - label_var_indices: shape (B,) or scalar — one target variable per batch item.
      Multi-target items require multiple calls.
    """
    if label_times is None and label_var_indices is None:
        return
    if label_times is None or label_var_indices is None:
        raise ValueError(
            "label_times and label_var_indices must either both be provided "
            "or both be None."
        )

    batch_size, num_steps, num_vars = obs_mask.shape
    label_t = np.asarray(label_times, dtype=np.int64)
    if label_t.ndim != 2 or label_t.shape[0] != batch_size:
        raise ValueError(
            "label_times must have shape (B, N_labels) with B matching obs_mask."
        )
    if label_t.min() < 0 or label_t.max() >= num_steps:
        raise ValueError("label_times must be in [0, T).")

    if np.isscalar(label_var_indices):
        label_vars = np.full((batch_size,), int(label_var_indices), dtype=np.int64)
    else:
        label_vars = np.asarray(label_var_indices, dtype=np.int64)
    if label_vars.shape != (batch_size,):
        raise ValueError("label_var_indices must have shape (B,) or be scalar.")
    if label_vars.min() < 0 or label_vars.max() >= num_vars:
        raise ValueError("label_var_indices must be in [0, p).")

    for batch_idx in range(batch_size):
        obs_mask[batch_idx, label_t[batch_idx], label_vars[batch_idx]] = True


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(clipped) - np.log1p(-clipped)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))
