"""Concise tests for DynSCM batch assembly."""

from __future__ import annotations

import numpy as np
import pytest
import torch

_REQUIRED_BATCH_KEYS = {
    "x",
    "y",
    "target_y",
    "single_eval_pos",
    "num_datapoints",
    "target_mask",
}
_RICHNESS_METADATA_KEYS = {
    "sampled_mechanism_type_id",
    "sampled_noise_family_id",
    "sampled_missing_mode_id",
    "sampled_kernel_family_id",
    "sampled_student_df",
    "sampled_num_vars",
    "sampled_num_steps",
    "sampled_n_train",
    "sampled_n_test",
    "sampled_pre_budget_feature_count",
    "sampled_simulation_clipped",
    "sampled_simulation_num_attempts",
    "sampled_simulation_max_abs_value",
    "sampled_train_target_std",
    "sampled_test_target_std",
    "sampled_max_abs_target_value",
    "sampled_probe_r2_train",
    "sampled_probe_r2_holdout",
    "sampled_probe_r2",
    "sampled_informative_feature_count",
    "sampled_informative_feature_std_floor",
    "sampled_target_parent_count_native",
    "sampled_target_parent_count_final",
    "sampled_target_parent_count",
    "sampled_target_self_lag_weight_native",
    "sampled_target_self_lag_weight_final",
    "sampled_target_self_lag_weight",
    "sampled_target_native_lag1_self_edge",
    "sampled_target_had_forced_lag_parent",
    "sampled_target_had_forced_self_lag",
    "sampled_mask_channels_enabled",
    "sampled_noise_scale",
    "sampled_missing_fraction",
    "sampled_block_missing_fraction",
    "sampled_generation_attempts_used",
    "sampled_low_std_reject_count",
    "sampled_probe_r2_reject_count",
    "sampled_clipped_reject_count",
    "sampled_informative_feature_reject_count",
    "sampled_missing_reject_count",
    "sampled_filter_accept",
}


def _close_if_supported(get_batch) -> None:
    close_fn = getattr(get_batch, "close", None)
    if callable(close_fn):
        close_fn()


@pytest.fixture(scope="module")
def dynscm_api(dynscm_modules):
    return dynscm_modules["config"], dynscm_modules["get_batch"]


def test_make_get_batch_dynscm_contract_shapes_and_padding(dynscm_api):
    config_mod, get_batch_mod = dynscm_api
    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "num_variables_min": 3,
            "num_variables_max": 3,
            "series_length_min": 80,
            "series_length_max": 80,
            "max_lag": 4,
            "mechanism_type": "linear_var",
            "missing_mode": "mix",
            "train_rows_min": 4,
            "train_rows_max": 4,
            "test_rows_min": 2,
            "test_rows_max": 2,
            "forecast_horizons": (1, 2, 3),
            "num_kernels": 1,
        }
    )
    get_batch = get_batch_mod.make_get_batch_dynscm(
        cfg,
        device=torch.device("cpu"),
        seed=7,
    )

    batch = get_batch(batch_size=3, num_datapoints_max=9, num_features=12)

    assert _REQUIRED_BATCH_KEYS.issubset(set(batch))
    assert _RICHNESS_METADATA_KEYS.issubset(set(batch))
    assert isinstance(batch["single_eval_pos"], int)
    assert isinstance(batch["num_datapoints"], int)

    x = batch["x"]
    y = batch["y"]
    target_y = batch["target_y"]
    target_mask = batch["target_mask"]

    assert x.shape == (3, 9, 12)
    assert y.shape == (3, 9)
    assert target_y.shape == (3, 9)
    assert batch["single_eval_pos"] == 4
    assert batch["num_datapoints"] == 6
    assert target_mask.shape == (3, 9)
    assert target_mask.dtype == torch.bool
    assert torch.all(target_mask[:, :4] == 0)
    assert torch.all(target_mask[:, 4:6] == 1)
    assert torch.all(target_mask[:, 6:] == 0)

    assert torch.equal(y, target_y)
    assert torch.isfinite(x).all()
    assert torch.isfinite(y).all()
    assert not torch.isnan(y[:, : batch["single_eval_pos"]]).any()

    # With fixed (n_train=4, n_test=2), tail rows [6:] must be zero-padded.
    assert torch.allclose(x[:, 6:, :], torch.zeros_like(x[:, 6:, :]))
    assert torch.allclose(y[:, 6:], torch.zeros_like(y[:, 6:]))


def test_make_get_batch_dynscm_is_deterministic_across_closures(dynscm_api):
    config_mod, get_batch_mod = dynscm_api
    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "num_variables_min": 3,
            "num_variables_max": 4,
            "series_length_min": 90,
            "series_length_max": 100,
            "max_lag": 5,
            "mechanism_type": "linear_var",
            "missing_mode": "mix",
            "train_rows_min": 6,
            "train_rows_max": 8,
            "test_rows_min": 3,
            "test_rows_max": 4,
        }
    )

    get_batch_a = get_batch_mod.make_get_batch_dynscm(
        cfg,
        device=torch.device("cpu"),
        seed=19,
    )
    get_batch_b = get_batch_mod.make_get_batch_dynscm(
        cfg,
        device=torch.device("cpu"),
        seed=19,
    )

    batch_a_1 = get_batch_a(batch_size=2, num_datapoints_max=20, num_features=24)
    batch_b_1 = get_batch_b(batch_size=2, num_datapoints_max=20, num_features=24)
    batch_a_2 = get_batch_a(batch_size=2, num_datapoints_max=20, num_features=24)
    batch_b_2 = get_batch_b(batch_size=2, num_datapoints_max=20, num_features=24)

    assert torch.equal(batch_a_1["x"], batch_b_1["x"])
    assert torch.equal(batch_a_1["y"], batch_b_1["y"])
    assert batch_a_1["single_eval_pos"] == batch_b_1["single_eval_pos"]
    assert batch_a_1["num_datapoints"] == batch_b_1["num_datapoints"]
    assert torch.equal(batch_a_1["target_mask"], batch_b_1["target_mask"])
    for key in _RICHNESS_METADATA_KEYS:
        assert torch.equal(batch_a_1[key], batch_b_1[key])

    assert torch.equal(batch_a_2["x"], batch_b_2["x"])
    assert torch.equal(batch_a_2["y"], batch_b_2["y"])
    assert batch_a_2["single_eval_pos"] == batch_b_2["single_eval_pos"]
    assert batch_a_2["num_datapoints"] == batch_b_2["num_datapoints"]
    assert torch.equal(batch_a_2["target_mask"], batch_b_2["target_mask"])
    for key in _RICHNESS_METADATA_KEYS:
        assert torch.equal(batch_a_2[key], batch_b_2[key])

    _close_if_supported(get_batch_a)
    _close_if_supported(get_batch_b)


def test_make_get_batch_dynscm_enforces_target_lag_and_mask_gating(dynscm_api):
    config_mod, get_batch_mod = dynscm_api
    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "num_variables_min": 3,
            "num_variables_max": 3,
            "series_length_min": 96,
            "series_length_max": 96,
            "max_lag": 4,
            "mechanism_type": "linear_var",
            "noise_family": "normal",
            "missing_mode": "off",
            "add_mask_channels": True,
            "train_rows_min": 6,
            "train_rows_max": 6,
            "test_rows_min": 3,
            "test_rows_max": 3,
            "enforce_target_lagged_parent": True,
            "force_target_self_lag_if_parentless": True,
            "target_self_lag_min_budget_fraction": 0.30,
            "learnability_probe": True,
            "informative_feature_std_floor": 1e-3,
            "min_informative_feature_count": 1,
        }
    )

    get_batch = get_batch_mod.make_get_batch_dynscm(
        cfg,
        device=torch.device("cpu"),
        seed=23,
    )
    try:
        batch = get_batch(batch_size=4, num_datapoints_max=12, num_features=20)
    finally:
        _close_if_supported(get_batch)

    assert torch.all(batch["sampled_target_parent_count_final"] >= 1)
    assert torch.all(batch["sampled_target_self_lag_weight_final"] > 0)
    assert torch.all(batch["sampled_mask_channels_enabled"] == 0)


def test_make_get_batch_dynscm_parallel_matches_serial(dynscm_api):
    config_mod, get_batch_mod = dynscm_api
    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "num_variables_min": 3,
            "num_variables_max": 4,
            "series_length_min": 90,
            "series_length_max": 100,
            "max_lag": 5,
            "mechanism_type": "linear_var",
            "missing_mode": "mix",
            "num_kernels": 1,
            "train_rows_min": 6,
            "train_rows_max": 8,
            "test_rows_min": 3,
            "test_rows_max": 4,
        }
    )

    serial = get_batch_mod.make_get_batch_dynscm(
        cfg,
        device=torch.device("cpu"),
        seed=41,
        workers=1,
        worker_blas_threads=1,
    )
    parallel = get_batch_mod.make_get_batch_dynscm(
        cfg,
        device=torch.device("cpu"),
        seed=41,
        workers=2,
        worker_blas_threads=1,
    )
    try:
        serial_1 = serial(batch_size=2, num_datapoints_max=20, num_features=24)
        parallel_1 = parallel(batch_size=2, num_datapoints_max=20, num_features=24)
        serial_2 = serial(batch_size=2, num_datapoints_max=20, num_features=24)
        parallel_2 = parallel(batch_size=2, num_datapoints_max=20, num_features=24)
    finally:
        _close_if_supported(serial)
        _close_if_supported(parallel)

    assert torch.equal(serial_1["x"], parallel_1["x"])
    assert torch.equal(serial_1["y"], parallel_1["y"])
    assert serial_1["single_eval_pos"] == parallel_1["single_eval_pos"]
    assert serial_1["num_datapoints"] == parallel_1["num_datapoints"]
    assert torch.equal(serial_1["target_mask"], parallel_1["target_mask"])
    for key in _RICHNESS_METADATA_KEYS:
        assert torch.equal(serial_1[key], parallel_1[key])

    assert torch.equal(serial_2["x"], parallel_2["x"])
    assert torch.equal(serial_2["y"], parallel_2["y"])
    assert serial_2["single_eval_pos"] == parallel_2["single_eval_pos"]
    assert serial_2["num_datapoints"] == parallel_2["num_datapoints"]
    assert torch.equal(serial_2["target_mask"], parallel_2["target_mask"])
    for key in _RICHNESS_METADATA_KEYS:
        assert torch.equal(serial_2[key], parallel_2[key])


def test_batch_shared_family_sampling_is_constant_within_batch(dynscm_api):
    config_mod, get_batch_mod = dynscm_api
    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "num_variables_min": 3,
            "num_variables_max": 4,
            "series_length_min": 96,
            "series_length_max": 120,
            "max_lag": 6,
            "train_rows_min": 8,
            "train_rows_max": 8,
            "test_rows_min": 4,
            "test_rows_max": 4,
            "mechanism_type_choices": ["linear_var", "linear_plus_residual"],
            "mechanism_type_probs": [0.5, 0.5],
            "noise_family_choices": ["normal", "student_t"],
            "noise_family_probs": [0.5, 0.5],
            "missing_mode_choices": ["off", "mcar"],
            "missing_mode_probs": [0.5, 0.5],
            "kernel_family_choices": ["exp_decay", "mix"],
            "kernel_family_probs": [0.5, 0.5],
        }
    )
    from tfmplayground.priors.dynscm.research import (
        sample_batch_shared_family_overrides,
    )

    get_batch = get_batch_mod.make_get_batch_dynscm(
        cfg,
        device=torch.device("cpu"),
        seed=13,
        cfg_override_sampler=lambda rng, batch_index: (
            sample_batch_shared_family_overrides(  # noqa: E501
                cfg,
                rng,
                family_fields=(
                    "mechanism_type",
                    "noise_family",
                    "missing_mode",
                    "kernel_family",
                ),
            )
        ),
    )
    batch = get_batch(batch_size=4, num_datapoints_max=16, num_features=24)

    for key in (
        "sampled_mechanism_type_id",
        "sampled_noise_family_id",
        "sampled_missing_mode_id",
        "sampled_kernel_family_id",
    ):
        assert torch.unique(batch[key]).numel() == 1


def test_parallel_matches_serial_with_batch_shared_overrides(dynscm_api):
    config_mod, get_batch_mod = dynscm_api
    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "num_variables_min": 3,
            "num_variables_max": 4,
            "series_length_min": 96,
            "series_length_max": 120,
            "max_lag": 6,
            "train_rows_min": 8,
            "train_rows_max": 8,
            "test_rows_min": 4,
            "test_rows_max": 4,
            "mechanism_type_choices": ["linear_var", "linear_plus_residual"],
            "mechanism_type_probs": [0.5, 0.5],
            "noise_family_choices": ["normal", "student_t"],
            "noise_family_probs": [0.5, 0.5],
        }
    )
    from tfmplayground.priors.dynscm.research import (
        sample_batch_shared_family_overrides,
    )

    sampler = lambda rng, batch_index: sample_batch_shared_family_overrides(  # noqa: E731
        cfg,
        rng,
        family_fields=("mechanism_type", "noise_family"),
    )
    serial = get_batch_mod.make_get_batch_dynscm(
        cfg,
        device=torch.device("cpu"),
        seed=41,
        workers=1,
        cfg_override_sampler=sampler,
    )
    parallel = get_batch_mod.make_get_batch_dynscm(
        cfg,
        device=torch.device("cpu"),
        seed=41,
        workers=2,
        worker_blas_threads=1,
        cfg_override_sampler=sampler,
    )
    try:
        batch_serial = serial(batch_size=3, num_datapoints_max=16, num_features=24)
        batch_parallel = parallel(batch_size=3, num_datapoints_max=16, num_features=24)
    finally:
        _close_if_supported(serial)
        _close_if_supported(parallel)

    for key, value in batch_serial.items():
        if torch.is_tensor(value):
            assert torch.equal(value, batch_parallel[key])
        else:
            assert value == batch_parallel[key]


def test_share_system_within_batch_pins_latent_system_structure(dynscm_api):
    config_mod, get_batch_mod = dynscm_api
    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "num_variables_min": 3,
            "num_variables_max": 8,
            "series_length_min": 96,
            "series_length_max": 140,
            "max_lag": 8,
            "train_rows_min": 8,
            "train_rows_max": 8,
            "test_rows_min": 4,
            "test_rows_max": 4,
            "mechanism_type_choices": ["linear_var", "linear_plus_residual"],
            "mechanism_type_probs": [0.5, 0.5],
            "noise_family_choices": ["normal", "student_t"],
            "noise_family_probs": [0.5, 0.5],
            "student_df_min": 3.5,
            "student_df_max": 8.0,
        }
    )
    get_batch = get_batch_mod.make_get_batch_dynscm(
        cfg,
        device=torch.device("cpu"),
        seed=71,
        share_system_within_batch=True,
    )
    batch = get_batch(batch_size=8, num_datapoints_max=16, num_features=24)

    assert "sampled_shared_system_seed" in batch
    assert torch.unique(batch["sampled_shared_system_seed"]).numel() == 1
    assert torch.unique(batch["sampled_num_vars"]).numel() == 1
    assert torch.unique(batch["sampled_mechanism_type_id"]).numel() == 1
    assert torch.unique(batch["sampled_noise_family_id"]).numel() == 1
    assert torch.unique(batch["sampled_student_df"]).numel() == 1


def test_filtered_generation_raises_after_bounded_resampling(dynscm_api):
    config_mod, _ = dynscm_api
    from tfmplayground.priors.dynscm import parallel as parallel_mod
    from tfmplayground.priors.dynscm.research import DynSCMSampleFilterConfig

    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "num_variables_min": 3,
            "num_variables_max": 4,
            "series_length_min": 96,
            "series_length_max": 120,
            "max_lag": 6,
            "train_rows_min": 8,
            "train_rows_max": 8,
            "test_rows_min": 4,
            "test_rows_max": 4,
            "learnability_probe": True,
        }
    )
    sample_filter = DynSCMSampleFilterConfig(min_train_target_std=1e9)

    with pytest.raises(
        RuntimeError, match="rejected after exhausting generation attempts"
    ):
        parallel_mod.build_single_dynscm_sample(
            cfg,
            sample_seed=77,
            n_train=8,
            n_test=4,
            row_budget=16,
            num_features=24,
            sample_filter=sample_filter,
            max_generation_attempts=2,
        )


def test_probe_metrics_expose_train_and_holdout_scores() -> None:
    from tfmplayground.priors.dynscm.difficulty import (
        ridge_holdout_predictions,
        ridge_probe_r2_holdout,
        ridge_probe_r2_train,
    )

    x = np.array([[0.0], [1.0], [2.0], [3.0], [100.0], [101.0]], dtype=np.float64)
    y = np.array([0.0, 1.0, 2.0, 3.0, -100.0, -101.0], dtype=np.float64)

    assert ridge_probe_r2_train(x, y, n_train=4) > 0.99
    assert ridge_probe_r2_holdout(x, y, n_train=4) < -0.9
    preds = ridge_holdout_predictions(x, y, n_train=4)
    assert preds.shape == (2,)
    assert preds[0] > 50.0
    assert preds[1] > 50.0


def test_filter_uses_train_probe_r2_over_holdout_alias() -> None:
    from tfmplayground.priors.dynscm.research import DynSCMSampleFilterConfig

    sample_filter = DynSCMSampleFilterConfig(min_probe_train_r2=0.9)
    metadata = {
        "sampled_probe_r2_train": 0.95,
        "sampled_probe_r2_holdout": -1.0,
        "sampled_probe_r2": -1.0,
    }

    assert sample_filter.rejection_reason(metadata) is None


def test_filter_legacy_probe_thresholds_fall_back_to_probe_alias() -> None:
    from tfmplayground.priors.dynscm.research import DynSCMSampleFilterConfig

    sample_filter = DynSCMSampleFilterConfig(min_probe_r2=0.5)
    assert sample_filter.rejection_reason({"sampled_probe_r2": 0.25}) == "probe_r2_low"


def test_filter_accept_metadata_is_set_only_for_returned_samples(dynscm_api):
    config_mod, _ = dynscm_api
    from tfmplayground.priors.dynscm import parallel as parallel_mod
    from tfmplayground.priors.dynscm.research import DynSCMSampleFilterConfig

    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "num_variables_min": 3,
            "num_variables_max": 3,
            "series_length_min": 96,
            "series_length_max": 96,
            "max_lag": 4,
            "mechanism_type": "linear_var",
            "noise_family": "normal",
            "missing_mode": "off",
            "train_rows_min": 6,
            "train_rows_max": 6,
            "test_rows_min": 3,
            "test_rows_max": 3,
            "learnability_probe": True,
        }
    )

    _, _, metadata = parallel_mod.build_single_dynscm_sample(
        cfg,
        sample_seed=101,
        n_train=6,
        n_test=3,
        row_budget=12,
        num_features=20,
        sample_filter=DynSCMSampleFilterConfig(
            min_probe_train_r2=0.0,
            max_probe_train_r2=1.1,
        ),
        max_generation_attempts=4,
    )

    assert metadata["sampled_filter_accept"] == 1
    assert metadata["sampled_generation_attempts_used"] >= 1


def test_generation_runtime_errors_are_retried(monkeypatch, dynscm_api):
    config_mod, _ = dynscm_api
    from tfmplayground.priors.dynscm import parallel as parallel_mod

    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "num_variables_min": 3,
            "num_variables_max": 3,
            "series_length_min": 80,
            "series_length_max": 80,
            "train_rows_min": 4,
            "train_rows_max": 4,
            "test_rows_min": 2,
            "test_rows_max": 2,
        }
    )
    attempts: list[int] = []

    def fake_once(
        cfg,
        *,
        sample_seed,
        n_train,
        n_test,
        row_budget,
        num_features,
    ):
        attempts.append(int(sample_seed))
        if len(attempts) == 1:
            raise RuntimeError("Spectral radius constraint violated.")
        return (
            np.full((row_budget, num_features), 2.0, dtype=np.float32),
            np.full((row_budget,), 3.0, dtype=np.float32),
            {
                "sampled_simulation_clipped": 0,
                "sampled_simulation_num_attempts": 1,
                "sampled_simulation_max_abs_value": 1.0,
                "sampled_train_target_std": 0.5,
                "sampled_filter_accept": 1,
            },
        )

    monkeypatch.setattr(parallel_mod, "_build_single_dynscm_sample_once", fake_once)

    x, y, metadata = parallel_mod.build_single_dynscm_sample(
        cfg,
        sample_seed=11,
        n_train=4,
        n_test=2,
        row_budget=8,
        num_features=6,
        max_generation_attempts=2,
    )

    assert len(attempts) == 2
    assert x.shape == (8, 6)
    assert y.shape == (8,)
    assert metadata["sampled_filter_accept"] == 1
    assert float(x[0, 0]) == 2.0
    assert float(y[0]) == 3.0


def test_feature_priority_truncation_order_is_deterministic(dynscm_api):
    _, get_batch_mod = dynscm_api

    x = np.arange(1 * 2 * 10, dtype=np.float64).reshape(1, 2, 10)
    slices = {
        "data_values": (0, 6),
        "lags_value": (0, 4),
        "kern_value": (4, 6),
        "deterministic": (6, 8),
        "data_mask": (8, 10),
    }
    prioritized = get_batch_mod._prioritize_feature_blocks(
        x,
        feature_slices=slices,
    )

    expected = np.concatenate(
        [
            x[:, :, 6:8],  # deterministic first
            x[:, :, 0:4],  # explicit lags
            x[:, :, 4:6],  # kernels
            x[:, :, 8:10],  # masks
        ],
        axis=2,
    )
    assert np.array_equal(prioritized, expected)

    truncated = get_batch_mod._fit_feature_budget(prioritized, num_features=5)
    assert truncated.shape == (1, 2, 5)
    assert np.array_equal(truncated, expected[:, :, :5])


def test_get_batch_validates_inputs(dynscm_api):
    config_mod, get_batch_mod = dynscm_api
    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "num_variables_min": 3,
            "num_variables_max": 3,
            "series_length_min": 80,
            "series_length_max": 80,
            "max_lag": 4,
            "mechanism_type": "linear_var",
            "train_rows_min": 4,
            "train_rows_max": 4,
            "test_rows_min": 2,
            "test_rows_max": 2,
        }
    )
    get_batch = get_batch_mod.make_get_batch_dynscm(
        cfg, device=torch.device("cpu"), seed=99
    )

    with pytest.raises(ValueError, match="batch_size"):
        get_batch(batch_size=0, num_datapoints_max=10, num_features=8)

    with pytest.raises(ValueError, match="num_datapoints_max"):
        get_batch(batch_size=1, num_datapoints_max=1, num_features=8)

    with pytest.raises(ValueError, match="num_features"):
        get_batch(batch_size=1, num_datapoints_max=10, num_features=0)

    # Use valid config but num_datapoints_max too small for any feasible pair.
    with pytest.raises(ValueError, match="feasible"):
        get_batch(batch_size=1, num_datapoints_max=2, num_features=8)

    with pytest.raises(ValueError, match="workers"):
        get_batch_mod.make_get_batch_dynscm(
            cfg,
            device=torch.device("cpu"),
            seed=99,
            workers=0,
        )


@pytest.mark.parametrize(
    (
        "missing_mode",
        "use_contemp_edges",
        "drift_std",
        "num_kernels",
        "add_mask_channels",
    ),
    [
        ("off", True, 0.0, 0, False),
        ("mcar", False, 0.03, 2, True),
        ("mar", True, 0.0, 2, False),
        ("mnar_lite", False, 0.02, 0, True),
        ("mix", True, 0.05, 2, True),
    ],
)
def test_get_batch_scenario_matrix(
    dynscm_api,
    missing_mode: str,
    use_contemp_edges: bool,
    drift_std: float,
    num_kernels: int,
    add_mask_channels: bool,
):
    config_mod, get_batch_mod = dynscm_api
    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "num_variables_min": 3,
            "num_variables_max": 4,
            "series_length_min": 96,
            "series_length_max": 96,
            "max_lag": 6,
            "mechanism_type": "linear_var",
            "use_contemp_edges": use_contemp_edges,
            "drift_std": drift_std,
            "missing_mode": missing_mode,
            "num_kernels": num_kernels,
            "add_mask_channels": add_mask_channels,
            "train_rows_min": 8,
            "train_rows_max": 8,
            "test_rows_min": 4,
            "test_rows_max": 4,
        }
    )
    get_batch = get_batch_mod.make_get_batch_dynscm(
        cfg,
        device=torch.device("cpu"),
        seed=123,
    )

    batch = get_batch(batch_size=2, num_datapoints_max=16, num_features=24)
    x = batch["x"]
    y = batch["y"]
    split = int(batch["single_eval_pos"])
    num_datapoints = int(batch["num_datapoints"])
    mask = batch["target_mask"]

    assert x.shape == (2, 16, 24)
    assert y.shape == (2, 16)
    assert 0 < split < 16
    assert split < num_datapoints <= 16
    assert mask.shape == (2, 16)
    assert _RICHNESS_METADATA_KEYS.issubset(set(batch))
    assert torch.all(mask[:, :split] == 0)
    assert torch.all(mask[:, split:num_datapoints] == 1)
    assert torch.all(mask[:, num_datapoints:] == 0)
    assert torch.isfinite(x).all()
    assert torch.isfinite(y).all()
    assert not torch.isnan(x).any()
    assert not torch.isnan(y[:, :split]).any()
