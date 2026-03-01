from __future__ import annotations


def test_research_profiles_define_expected_names(priors_modules) -> None:
    profiles_mod = priors_modules["research_profiles"]
    assert profiles_mod.list_research_profiles() == (
        "medium32k_live_baseline",
        "medium32k_live_guardrails",
        "medium32k_live_batch_homogeneous",
        "medium32k_live_mode_ladder",
        "medium32k_live_mixture",
        "temporal_length_only_16k",
        "temporal_regimes_only_16k",
        "temporal_drift_only_16k",
        "temporal_regimes_plus_drift_16k",
        "temporal_length_plus_regimes_16k",
        "temporal_full_medium32k_reference",
        "benchmark_contract_observed_easy",
        "benchmark_contract_observed_temporal",
        "mode_ladder_norm_none",
        "mode_ladder_norm_zscore",
        "mode_ladder_norm_clamped",
        "integration_contract_easy",
        "integration_contract_temporal",
    )


def test_temporal_cfg_preserves_only_temporal_changes_vs_easy_family_stable(
    priors_modules,
) -> None:
    profiles_mod = priors_modules["research_profiles"]
    stable_cfg = profiles_mod.stable_cfg()
    temporal_cfg = profiles_mod.temporal_cfg()
    stable_easy = stable_cfg.with_overrides(
        mechanism_type="linear_var",
        mechanism_type_choices=None,
        mechanism_type_probs=None,
        noise_family="normal",
        noise_family_choices=None,
        noise_family_probs=None,
        missing_mode="off",
        missing_mode_choices=None,
        missing_mode_probs=None,
        kernel_family="exp_decay",
        kernel_family_choices=None,
        kernel_family_probs=None,
    )

    changed_fields = {
        key
        for key, value in temporal_cfg.to_dict().items()
        if stable_easy.to_dict()[key] != value
    }
    assert changed_fields == {
        "num_regimes",
        "sticky_rho",
        "drift_std",
        "series_length_min",
        "series_length_max",
    }


def test_research_profile_sources_and_suites_are_resolved(priors_modules) -> None:
    profiles_mod = priors_modules["research_profiles"]
    profile = profiles_mod.get_research_profile("medium32k_live_mixture")

    assert profile.train_source.kind == "mixture"
    assert tuple(name for name, _cfg in profile.train_source.child_sources) == (
        "stable_cfg",
        "temporal_cfg",
        "target_cfg",
    )
    assert profile.val_source.kind == "single"
    assert tuple(suite.name for suite in profile.eval_suites) == (
        "stable_eval",
        "temporal_eval",
        "temporal_eval_hard",
        "target_eval",
        "full_eval",
    )


def test_temporal_ablation_profiles_change_only_intended_temporal_axes(
    priors_modules,
) -> None:
    profiles_mod = priors_modules["research_profiles"]
    base = profiles_mod.temporal_learnable_cfg()

    length_only = profiles_mod.get_research_profile(
        "temporal_length_only_16k"
    ).train_source.cfg
    assert length_only.series_length_max == 256
    assert length_only.num_regimes == 1
    assert length_only.drift_std == 0.0
    assert length_only.noise_family == base.noise_family

    regimes_only = profiles_mod.get_research_profile(
        "temporal_regimes_only_16k"
    ).train_source.cfg
    assert regimes_only.series_length_max == 128
    assert regimes_only.num_regimes == 2
    assert regimes_only.drift_std == 0.0

    drift_only = profiles_mod.get_research_profile(
        "temporal_drift_only_16k"
    ).train_source.cfg
    assert drift_only.series_length_max == 128
    assert drift_only.num_regimes == 1
    assert drift_only.drift_std == 0.01

    full_ref = profiles_mod.get_research_profile(
        "temporal_full_medium32k_reference"
    ).train_source.cfg
    assert full_ref.series_length_max == 256
    assert full_ref.num_regimes == 2
    assert full_ref.drift_std == 0.01


def test_benchmark_contract_profiles_match_benchmark_shape_contract(
    priors_modules,
) -> None:
    profiles_mod = priors_modules["research_profiles"]
    expected_series = {
        "benchmark_contract_observed_easy": (64, 128),
        "benchmark_contract_observed_temporal": (128, 256),
    }
    for name in expected_series:
        cfg = profiles_mod.get_research_profile(name).train_source.cfg
        assert cfg.num_variables_min == 2
        assert cfg.num_variables_max == 2
        assert cfg.train_rows_min == 32
        assert cfg.train_rows_max == 32
        assert cfg.test_rows_min == 16
        assert cfg.test_rows_max == 16
        assert cfg.forecast_horizons == (1, 3)
        assert cfg.explicit_lags == (0, 1, 2, 5, 10)
        assert cfg.num_kernels == 3
        assert cfg.add_mask_channels is False
        assert cfg.missing_mode == "off"
        assert cfg.mechanism_type == "linear_var"
        assert cfg.noise_family == "normal"
        assert cfg.kernel_family == "exp_decay"
        assert cfg.enforce_target_lagged_parent is True
        assert cfg.learnability_probe is True
        assert (cfg.series_length_min, cfg.series_length_max) == expected_series[name]


def test_revised_research_profiles_use_only_short_horizons(priors_modules) -> None:
    profiles_mod = priors_modules["research_profiles"]
    for name in (
        "medium32k_live_baseline",
        "medium32k_live_guardrails",
        "medium32k_live_batch_homogeneous",
        "medium32k_live_mode_ladder",
        "medium32k_live_mixture",
        *profiles_mod.TEMPORAL_ABLATION_PROFILES,
    ):
        profile = profiles_mod.get_research_profile(name)
        if profile.train_source.cfg is not None:
            assert profile.train_source.cfg.forecast_horizons == (1, 3)
        if profile.val_source.cfg is not None:
            assert profile.val_source.cfg.forecast_horizons == (1, 3)


def test_temporal_profiles_use_shorter_common_budget(priors_modules) -> None:
    profiles_mod = priors_modules["research_profiles"]
    for name in profiles_mod.TEMPORAL_ABLATION_PROFILES:
        profile = profiles_mod.get_research_profile(name)
        assert profile.training_budget.epochs == 12
        assert profile.training_budget.steps == 400


def test_target_learnable_cfg_uses_native_separated_sampler(priors_modules) -> None:
    profiles_mod = priors_modules["research_profiles"]
    cfg = profiles_mod.target_learnable_cfg()

    assert cfg.lagged_sampler_mode == "separated_self_cross"
    assert cfg.self_lag_prob == 0.80
    assert cfg.self_lag_decay_rate == 0.15
    assert cfg.base_lagged_edge_prob == 0.40
    assert cfg.target_self_lag_magnitude_min == 0.55
    assert cfg.target_self_lag_magnitude_max == 0.85
    assert cfg.force_positive_self_lag is True


def test_mode_ladder_profile_uses_softer_self_lag_floor_than_generic_target(
    priors_modules,
) -> None:
    profiles_mod = priors_modules["research_profiles"]
    profile = profiles_mod.get_research_profile("medium32k_live_mode_ladder")

    assert profile.train_source.cfg is not None
    assert profile.train_source.cfg.lagged_sampler_mode == "separated_self_cross"
    assert profile.train_source.cfg.target_self_lag_magnitude_min == 0.50
    assert profile.train_source.cfg.target_self_lag_magnitude_max == 0.78
    assert profile.train_source.cfg.target_self_lag_min_budget_fraction == 0.15
    assert profile.train_source.cfg.target_self_lag_abs_min == 0.10


def test_non_baseline_profiles_validate_on_raw_target_cfg(priors_modules) -> None:
    profiles_mod = priors_modules["research_profiles"]
    target_cfg = profiles_mod.target_learnable_cfg()

    for name in (
        "medium32k_live_guardrails",
        "medium32k_live_batch_homogeneous",
        "medium32k_live_mode_ladder",
        "medium32k_live_mixture",
    ):
        profile = profiles_mod.get_research_profile(name)
        assert profile.val_source.cfg == target_cfg


def test_normalization_ablation_profiles_only_change_target_normalization(
    priors_modules,
) -> None:
    profiles_mod = priors_modules["research_profiles"]
    assert (
        profiles_mod.get_research_profile("mode_ladder_norm_none").target_normalization
        == "none"
    )
    assert (
        profiles_mod.get_research_profile(
            "mode_ladder_norm_zscore"
        ).target_normalization
        == "per_function_zscore"
    )
    assert (
        profiles_mod.get_research_profile(
            "mode_ladder_norm_clamped"
        ).target_normalization
        == "per_function_clamped"
    )


def test_guardrailed_profiles_use_elevated_retry_budget(priors_modules) -> None:
    profiles_mod = priors_modules["research_profiles"]
    expected = profiles_mod.GUARDRAIL_MAX_SAMPLE_ATTEMPTS

    for name in (
        "medium32k_live_guardrails",
        "medium32k_live_batch_homogeneous",
        "medium32k_live_mode_ladder",
        "medium32k_live_mixture",
    ):
        profile = profiles_mod.get_research_profile(name)
        assert profile.train_source.max_sample_attempts_per_item == expected


def test_research_profiles_default_to_nonzero_weight_decay_and_stronger_self_lag(
    priors_modules,
) -> None:
    profiles_mod = priors_modules["research_profiles"]
    profile = profiles_mod.get_research_profile("medium32k_live_baseline")

    assert profile.training_budget.weight_decay == 1e-4
    assert profile.train_source.cfg is not None
    assert profile.train_source.cfg.lagged_sampler_mode == "separated_self_cross"
    assert profile.train_source.cfg.target_self_lag_magnitude_min == 0.55
    assert profile.train_source.cfg.target_self_lag_magnitude_max == 0.85
    assert profile.train_source.cfg.target_self_lag_min_budget_fraction == 0.25
    assert profile.train_source.cfg.target_self_lag_abs_min == 0.15
    assert profile.train_source.cfg.noise_scale_schedule_tag is None


def test_surviving_curriculum_profiles_use_shorter_early_series_windows(
    priors_modules,
) -> None:
    profiles_mod = priors_modules["research_profiles"]

    mode_ladder = profiles_mod.get_research_profile("medium32k_live_mode_ladder")
    mode_map = {
        mode.name: mode.cfg_overrides for mode in mode_ladder.train_source.modes
    }
    assert mode_map["graph_only"]["series_length_min"] == 64
    assert mode_map["graph_only"]["series_length_max"] == 128
    assert mode_map["noise_only"]["noise_family"] == "normal"
    assert mode_map["missing_only"]["missing_mode"] == "mcar"
    assert mode_map["missing_only"]["missing_mode_probs"] == (0.9, 0.1)
    assert mode_map["missing_only"]["missing_rate_min"] == 0.02
    assert mode_map["missing_only"]["missing_rate_max"] == 0.08
    assert mode_map["missing_only"]["block_missing_prob"] == 0.0
    assert mode_map["temporal_only"]["series_length_min"] == 96
    assert mode_map["temporal_only"]["series_length_max"] == 128
    assert mode_ladder.train_source.sample_filter is not None
    assert mode_ladder.train_source.sample_filter.min_probe_train_r2 == 0.08
    assert mode_ladder.train_source.sample_filter.max_missing_fraction == 0.18

    mixture = profiles_mod.get_research_profile("medium32k_live_mixture")
    child_sources = {name: cfg for name, cfg in mixture.train_source.child_sources}
    assert child_sources["stable_cfg"].series_length_min == 64
    assert child_sources["stable_cfg"].series_length_max == 128
    assert child_sources["temporal_cfg"].series_length_min == 96
    assert child_sources["temporal_cfg"].series_length_max == 128


def test_build_promotion_profile_upgrades_training_source_to_full_cfg(
    priors_modules,
) -> None:
    profiles_mod = priors_modules["research_profiles"]
    promotion = profiles_mod.build_promotion_profile("medium32k_live_mixture")

    assert promotion.name == "medium32k_live_mixture_full_promotion"
    assert tuple(name for name, _cfg in promotion.train_source.child_sources) == (
        "stable_cfg",
        "temporal_cfg",
        "full_cfg",
    )
