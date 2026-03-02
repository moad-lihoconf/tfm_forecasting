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
        "benchmark_contract_observed_easy_matched_val",
        "benchmark_contract_observed_temporal",
        "mode_ladder_norm_none",
        "mode_ladder_norm_zscore",
        "mode_ladder_norm_clamped",
        "integration_contract_easy",
        "integration_contract_easy_stable_batch",
        "integration_contract_easy_stable_batch_safe_eval",
        "integration_contract_easy_stable_batch_k4_safe_eval",
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
        assert cfg.noise_scale_min == 0.08
        assert cfg.noise_scale_max == 0.08
        assert cfg.noise_scale_schedule_tag is None
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


def test_mode_ladder_temporal_only_mode_disables_hidden_regime_and_drift_shift(
    priors_modules,
) -> None:
    profiles_mod = priors_modules["research_profiles"]
    profile = profiles_mod.get_research_profile("medium32k_live_mode_ladder")
    temporal_only = next(
        mode for mode in profile.train_source.modes if mode.name == "temporal_only"
    )

    assert temporal_only.cfg_overrides["num_regimes"] == 1
    assert temporal_only.cfg_overrides["sticky_rho"] == 1.0
    assert temporal_only.cfg_overrides["shared_order"] is True
    assert temporal_only.cfg_overrides["share_base_graph"] is True
    assert temporal_only.cfg_overrides["drift_std"] == 0.0


def test_mode_ladder_v2_adds_incremental_complexity_modes(priors_modules) -> None:
    profiles_mod = priors_modules["research_profiles"]
    profile = profiles_mod.get_research_profile("medium32k_live_mode_ladder")
    mode_map = {mode.name: mode.cfg_overrides for mode in profile.train_source.modes}

    assert tuple(mode_map) == (
        "graph_only",
        "temporal_only",
        "kernels_only",
        "missing_easy",
        "residual_easy",
        "regimes_easy",
        "full",
    )
    assert mode_map["graph_only"]["num_kernels"] == 0
    assert mode_map["graph_only"]["add_seasonality"] is False
    assert mode_map["kernels_only"]["num_kernels"] == 1
    assert mode_map["missing_easy"]["missing_mode"] == "mcar"
    assert mode_map["missing_easy"]["missing_rate_min"] == 0.01
    assert mode_map["missing_easy"]["missing_rate_max"] == 0.06
    assert mode_map["missing_easy"]["block_missing_prob"] == 0.0
    assert mode_map["residual_easy"]["mechanism_type_choices"] == (
        "linear_var",
        "linear_plus_residual",
    )
    assert mode_map["residual_easy"]["mechanism_type_probs"] == (0.8, 0.2)
    assert mode_map["residual_easy"]["residual_lipschitz_max"] == 0.03
    assert mode_map["regimes_easy"]["num_regimes"] == 2
    assert mode_map["regimes_easy"]["drift_std"] == 0.005


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


def test_benchmark_contract_easy_uses_stricter_high_signal_filter(
    priors_modules,
) -> None:
    profiles_mod = priors_modules["research_profiles"]
    profile = profiles_mod.get_research_profile("benchmark_contract_observed_easy")

    assert profile.train_source.sample_filter is not None
    assert profile.train_source.sample_filter.min_probe_train_r2 == 0.15
    assert profile.train_source.sample_filter.max_probe_train_r2 == 0.95
    assert profile.train_source.sample_filter.min_informative_feature_count == 8
    assert profile.val_source.sample_filter is None


def test_benchmark_contract_easy_matched_val_uses_strict_filter_for_both_splits(
    priors_modules,
) -> None:
    profiles_mod = priors_modules["research_profiles"]
    profile = profiles_mod.get_research_profile(
        "benchmark_contract_observed_easy_matched_val"
    )

    assert profile.train_source.sample_filter is not None
    assert profile.train_source.sample_filter.min_probe_train_r2 == 0.15
    assert profile.train_source.sample_filter.max_probe_train_r2 == 0.95
    assert profile.train_source.sample_filter.min_informative_feature_count == 8
    assert profile.val_source.sample_filter is not None
    assert profile.val_source.sample_filter.min_probe_train_r2 == 0.15
    assert profile.val_source.sample_filter.max_probe_train_r2 == 0.95
    assert profile.val_source.sample_filter.min_informative_feature_count == 8
    assert (
        profile.val_source.max_sample_attempts_per_item
        == profiles_mod.GUARDRAIL_MAX_SAMPLE_ATTEMPTS
    )


def test_integration_contract_easy_stable_batch_shares_latent_system_within_batch(
    priors_modules,
) -> None:
    profiles_mod = priors_modules["research_profiles"]
    profile = profiles_mod.get_research_profile(
        "integration_contract_easy_stable_batch"
    )

    assert profile.train_source.share_system_within_batch is True
    assert profile.val_source.share_system_within_batch is False
    assert profile.train_source.shared_system_reuse_batches == 1
    assert profile.val_source.shared_system_reuse_batches == 1
    assert profile.train_source.generation_exhaustion_policy == "raise"
    assert profile.val_source.generation_exhaustion_policy == "raise"
    assert profile.train_source.sample_filter is not None
    assert profile.val_source.sample_filter is not None
    assert profile.train_source.sample_filter.min_probe_train_r2 == 0.15
    assert profile.val_source.sample_filter.min_probe_train_r2 == 0.15
    assert profile.train_source.sample_filter.max_probe_train_r2 == 0.99
    assert profile.val_source.sample_filter.max_probe_train_r2 == 0.99


def test_integration_contract_easy_stable_batch_safe_eval_relaxes_val_and_soft_fails(
    priors_modules,
) -> None:
    profiles_mod = priors_modules["research_profiles"]
    profile = profiles_mod.get_research_profile(
        "integration_contract_easy_stable_batch_safe_eval"
    )

    assert profile.train_source.generation_exhaustion_policy == "raise"
    assert profile.val_source.generation_exhaustion_policy == "accept_last"
    assert profile.train_source.sample_filter is not None
    assert profile.val_source.sample_filter is not None
    assert profile.train_source.sample_filter.max_probe_train_r2 == 0.99
    assert profile.val_source.sample_filter.max_probe_train_r2 == 0.995
    assert profile.train_source.share_system_within_batch is True
    assert profile.val_source.share_system_within_batch is False
    assert profile.train_source.shared_system_reuse_batches == 1
    assert profile.val_source.shared_system_reuse_batches == 1


def test_integration_contract_easy_stable_batch_k4_safe_eval_reuses_system_k_batches(
    priors_modules,
) -> None:
    profiles_mod = priors_modules["research_profiles"]
    profile = profiles_mod.get_research_profile(
        "integration_contract_easy_stable_batch_k4_safe_eval"
    )

    assert profile.train_source.share_system_within_batch is True
    assert profile.train_source.shared_system_reuse_batches == 4
    assert profile.train_source.generation_exhaustion_policy == "raise"
    assert profile.val_source.generation_exhaustion_policy == "accept_last"
    assert profile.val_source.share_system_within_batch is False
    assert profile.val_source.shared_system_reuse_batches == 1


def test_normalization_ablation_profiles_only_change_target_normalization(
    priors_modules,
) -> None:
    profiles_mod = priors_modules["research_profiles"]
    assert (
        profiles_mod.get_research_profile(
            "medium32k_live_mode_ladder"
        ).target_normalization
        == "per_function_clamped"
    )
    assert (
        profiles_mod.get_research_profile("medium32k_live_mode_ladder").target_std_floor
        == 5e-2
    )
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
    assert mode_map["kernels_only"]["num_kernels"] == 1
    assert mode_map["missing_easy"]["missing_mode"] == "mcar"
    assert mode_map["missing_easy"]["missing_mode_choices"] is None
    assert mode_map["missing_easy"]["missing_rate_min"] == 0.01
    assert mode_map["missing_easy"]["missing_rate_max"] == 0.06
    assert mode_map["missing_easy"]["block_missing_prob"] == 0.0
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


def test_mode_ladder_schedule_widens_complexity_gradually(priors_modules) -> None:
    profiles_mod = priors_modules["research_profiles"]
    schedule = profiles_mod.get_research_profile(
        "medium32k_live_mode_ladder"
    ).train_source.schedule

    assert schedule is not None
    early = tuple(round(v, 4) for v in schedule.weights_at(0.10))
    middle = tuple(round(v, 4) for v in schedule.weights_at(0.65))
    late = tuple(round(v, 4) for v in schedule.weights_at(0.95))

    assert early == (0.4, 0.35, 0.25, 0.0, 0.0, 0.0, 0.0)
    assert middle[3] > 0.0
    assert middle[4] > 0.0
    assert late[5] > 0.0
    assert late[6] > 0.0


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
