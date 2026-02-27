"""Tests for per-sample DynSCM family sampling richness controls."""

from __future__ import annotations

import numpy as np
import pytest


def test_variant_choice_distribution_validation_rejects_invalid_payloads(
    dynscm_modules,
):
    config_mod = dynscm_modules["config"]

    with pytest.raises(ValueError, match="must both be provided"):
        config_mod.DynSCMConfig.from_dict(
            {"mechanism_type_choices": ["linear_var", "linear_plus_residual"]}
        )

    with pytest.raises(ValueError, match="identical lengths"):
        config_mod.DynSCMConfig.from_dict(
            {
                "noise_family_choices": ["normal", "student_t"],
                "noise_family_probs": [1.0],
            }
        )

    with pytest.raises(ValueError, match="must not contain duplicates"):
        config_mod.DynSCMConfig.from_dict(
            {
                "kernel_family_choices": ["mix", "mix"],
                "kernel_family_probs": [0.5, 0.5],
            }
        )

    with pytest.raises(ValueError, match="student_df_min and student_df_max"):
        config_mod.DynSCMConfig.from_dict({"student_df_min": 3.0})


def test_sample_dynscm_variant_cfg_is_seed_deterministic(dynscm_modules):
    config_mod = dynscm_modules["config"]
    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "mechanism_type_choices": ["linear_var", "linear_plus_residual"],
            "mechanism_type_probs": [0.35, 0.65],
            "noise_family_choices": ["normal", "student_t"],
            "noise_family_probs": [0.4, 0.6],
            "student_df_min": 3.5,
            "student_df_max": 8.0,
            "missing_mode_choices": ["off", "mcar", "mar", "mnar_lite", "mix"],
            "missing_mode_probs": [0.05, 0.2, 0.2, 0.2, 0.35],
            "kernel_family_choices": ["exp_decay", "power_law", "mix"],
            "kernel_family_probs": [0.25, 0.25, 0.5],
        }
    )

    rng_a = np.random.default_rng(19)
    rng_b = np.random.default_rng(19)
    trace_a = []
    trace_b = []
    for _ in range(64):
        variant_cfg_a, metadata_a = config_mod.sample_dynscm_variant_cfg(cfg, rng_a)
        variant_cfg_b, metadata_b = config_mod.sample_dynscm_variant_cfg(cfg, rng_b)
        trace_a.append(
            (
                variant_cfg_a.mechanism_type,
                variant_cfg_a.noise_family,
                variant_cfg_a.missing_mode,
                variant_cfg_a.kernel_family,
                round(float(metadata_a["sampled_student_df"]), 6),
            )
        )
        trace_b.append(
            (
                variant_cfg_b.mechanism_type,
                variant_cfg_b.noise_family,
                variant_cfg_b.missing_mode,
                variant_cfg_b.kernel_family,
                round(float(metadata_b["sampled_student_df"]), 6),
            )
        )

    assert trace_a == trace_b


def test_variant_sampling_covers_all_configured_families_and_df_bounds(dynscm_modules):
    config_mod = dynscm_modules["config"]
    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "mechanism_type_choices": ["linear_var", "linear_plus_residual"],
            "mechanism_type_probs": [1.0, 1.0],
            "noise_family_choices": ["normal", "student_t"],
            "noise_family_probs": [1.0, 1.0],
            "student_df_min": 3.5,
            "student_df_max": 8.0,
            "missing_mode_choices": ["off", "mcar", "mar", "mnar_lite", "mix"],
            "missing_mode_probs": [1.0, 1.0, 1.0, 1.0, 1.0],
            "kernel_family_choices": ["exp_decay", "power_law", "mix"],
            "kernel_family_probs": [1.0, 1.0, 1.0],
        }
    )

    rng = np.random.default_rng(7)
    mechanism_ids: set[int] = set()
    noise_ids: set[int] = set()
    missing_ids: set[int] = set()
    kernel_ids: set[int] = set()
    sampled_student_dfs: list[float] = []
    for _ in range(4000):
        _variant_cfg, metadata = config_mod.sample_dynscm_variant_cfg(cfg, rng)
        mechanism_ids.add(int(metadata["sampled_mechanism_type_id"]))
        noise_id = int(metadata["sampled_noise_family_id"])
        noise_ids.add(noise_id)
        missing_ids.add(int(metadata["sampled_missing_mode_id"]))
        kernel_ids.add(int(metadata["sampled_kernel_family_id"]))
        if noise_id == 1:  # student_t id
            sampled_student_dfs.append(float(metadata["sampled_student_df"]))

    assert mechanism_ids == {0, 1}
    assert noise_ids == {0, 1}
    assert missing_ids == {0, 1, 2, 3, 4}
    assert kernel_ids == {0, 1, 2}
    assert sampled_student_dfs
    assert min(sampled_student_dfs) >= 3.5
    assert max(sampled_student_dfs) <= 8.0
