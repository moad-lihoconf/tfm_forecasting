"""Concise tests for DynSCM raw missingness masks."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def dynscm_api(dynscm_modules):
    return dynscm_modules["config"], dynscm_modules["missingness"]


def _max_false_run(mask_1d: np.ndarray) -> int:
    best = 0
    current = 0
    for observed in mask_1d:
        if not observed:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def test_off_mode_returns_all_observed_boolean_mask(dynscm_api):
    config_mod, missing_mod = dynscm_api
    cfg = config_mod.DynSCMConfig.from_dict({"missing_mode": "off"})
    series = np.random.default_rng(0).normal(size=(4, 32, 5))

    obs_mask = missing_mod.sample_observation_mask(cfg, series, seed=1)
    assert obs_mask.shape == (4, 32, 5)
    assert obs_mask.dtype == np.bool_
    assert obs_mask.all()


@pytest.mark.parametrize("mode", ["mcar", "mar", "mnar_lite", "mix"])
def test_stochastic_modes_apply_block_outages_per_variable(dynscm_api, mode: str):
    config_mod, missing_mod = dynscm_api
    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "missing_mode": mode,
            "missing_rate_min": 0.10,
            "missing_rate_max": 0.10,
            "block_missing_prob": 1.0,
            "block_len_min": 4,
            "block_len_max": 4,
        }
    )
    series = np.random.default_rng(2).normal(size=(8, 40, 3))
    obs_mask = missing_mod.sample_observation_mask(cfg, series, seed=3)

    assert obs_mask.dtype == np.bool_
    assert obs_mask.shape == (8, 40, 3)
    for batch_idx in range(8):
        for var_idx in range(3):
            assert _max_false_run(obs_mask[batch_idx, :, var_idx]) >= 4


def test_mar_mode_uses_normalized_time_signal(dynscm_api):
    config_mod, missing_mod = dynscm_api
    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "missing_mode": "mar",
            "missing_rate_min": 0.20,
            "missing_rate_max": 0.20,
            "block_missing_prob": 0.0,
        }
    )
    series = np.zeros((256, 80, 1), dtype=np.float64)
    obs_mask = missing_mod.sample_observation_mask(cfg, series, seed=4)

    miss_early = (~obs_mask[:, :20, 0]).mean()
    miss_late = (~obs_mask[:, -20:, 0]).mean()
    assert miss_late > miss_early + 0.05


def test_mnar_lite_mode_depends_on_current_latent_magnitude(dynscm_api):
    config_mod, missing_mod = dynscm_api
    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "missing_mode": "mnar_lite",
            "missing_rate_min": 0.10,
            "missing_rate_max": 0.10,
            "block_missing_prob": 0.0,
        }
    )

    series = np.zeros((256, 60, 1), dtype=np.float64)
    series[:, :30, 0] = 0.01
    series[:, 30:, 0] = 8.0
    obs_mask = missing_mod.sample_observation_mask(cfg, series, seed=5)

    miss_low_mag = (~obs_mask[:, :30, 0]).mean()
    miss_high_mag = (~obs_mask[:, 30:, 0]).mean()
    assert miss_high_mag > miss_low_mag + 0.08


def test_sample_observation_mask_is_seed_deterministic(dynscm_api):
    config_mod, missing_mod = dynscm_api
    cfg = config_mod.DynSCMConfig.from_dict(
        {"missing_mode": "mix", "block_missing_prob": 0.5}
    )
    series = np.random.default_rng(8).normal(size=(4, 48, 3))
    mask_a = missing_mod.sample_observation_mask(cfg, series, seed=9)
    mask_b = missing_mod.sample_observation_mask(cfg, series, seed=9)
    assert np.array_equal(mask_a, mask_b)

    mask_c = missing_mod.sample_observation_mask(cfg, series, seed=10)
    assert not np.array_equal(mask_a, mask_c)


def test_sample_observation_mask_validates_inputs(dynscm_api):
    config_mod, missing_mod = dynscm_api
    cfg = config_mod.DynSCMConfig.from_dict({"missing_mode": "mcar"})

    with pytest.raises(ValueError, match="shape"):
        missing_mod.sample_observation_mask(cfg, np.zeros((4, 32)), seed=0)

    with pytest.raises(ValueError, match="missing_mode"):
        bad_cfg = config_mod.DynSCMConfig.from_dict({})
        bad_cfg = bad_cfg.model_copy(
            update={
                "missingness": bad_cfg.missingness.model_copy(
                    update={"missing_mode": "bad_mode"}
                )
            }
        )
        missing_mod.sample_observation_mask(bad_cfg, np.zeros((2, 10, 3)), seed=0)

    series = np.zeros((2, 10, 3))
    label_times = np.zeros((2, 1), dtype=np.int64)

    with pytest.raises(ValueError, match="both be provided"):
        missing_mod.sample_observation_mask(
            cfg, series, seed=0, label_times=label_times
        )

    with pytest.raises(ValueError, match="both be provided"):
        missing_mod.sample_observation_mask(
            cfg, series, seed=0, label_var_indices=np.array([0, 0])
        )

    out_of_bounds_times = np.full((2, 1), fill_value=99, dtype=np.int64)
    with pytest.raises(ValueError, match="label_times must be in"):
        missing_mod.sample_observation_mask(
            cfg,
            series,
            seed=0,
            label_times=out_of_bounds_times,
            label_var_indices=np.array([0, 0]),
        )

    out_of_bounds_vars = np.array([0, 99], dtype=np.int64)
    with pytest.raises(ValueError, match="label_var_indices must be in"):
        missing_mod.sample_observation_mask(
            cfg,
            series,
            seed=0,
            label_times=label_times,
            label_var_indices=out_of_bounds_vars,
        )


def test_label_positions_are_never_masked_when_provided(dynscm_api):
    config_mod, missing_mod = dynscm_api
    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "missing_mode": "mix",
            "missing_rate_min": 0.35,
            "missing_rate_max": 0.35,
            "block_missing_prob": 1.0,
            "block_len_min": 6,
            "block_len_max": 6,
        }
    )

    batch_size, num_steps, num_vars = 4, 36, 3
    series = np.random.default_rng(6).normal(size=(batch_size, num_steps, num_vars))
    label_times = np.array(
        [
            [4, 8, 12, 16],
            [5, 9, 13, 17],
            [6, 10, 14, 18],
            [7, 11, 15, 19],
        ],
        dtype=np.int64,
    )
    label_var_indices = np.array([0, 1, 2, 0], dtype=np.int64)

    obs_mask = missing_mod.sample_observation_mask(
        cfg,
        series,
        seed=7,
        label_times=label_times,
        label_var_indices=label_var_indices,
    )
    for batch_idx in range(batch_size):
        protected_values = obs_mask[
            batch_idx,
            label_times[batch_idx],
            label_var_indices[batch_idx],
        ]
        assert protected_values.all()
