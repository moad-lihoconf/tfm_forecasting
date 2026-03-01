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
    stable = profiles_mod.stable_cfg().to_dict()
    expected = {
        "temporal_length_only_16k": {"series_length_max"},
        "temporal_regimes_only_16k": {"num_regimes", "sticky_rho"},
        "temporal_drift_only_16k": {"drift_std"},
        "temporal_regimes_plus_drift_16k": {"num_regimes", "sticky_rho", "drift_std"},
        "temporal_length_plus_regimes_16k": {
            "series_length_max",
            "num_regimes",
            "sticky_rho",
        },
        "temporal_full_medium32k_reference": {
            "series_length_max",
            "num_regimes",
            "sticky_rho",
            "drift_std",
        },
    }
    for name, changed_keys in expected.items():
        profile = profiles_mod.get_research_profile(name)
        current = profile.train_source.cfg.to_dict()
        actual = {key for key, value in current.items() if stable[key] != value}
        assert actual == changed_keys


def test_benchmark_contract_profiles_match_benchmark_shape_contract(
    priors_modules,
) -> None:
    profiles_mod = priors_modules["research_profiles"]
    for name in (
        "benchmark_contract_observed_easy",
        "benchmark_contract_observed_temporal",
    ):
        cfg = profiles_mod.get_research_profile(name).train_source.cfg
        assert cfg.num_variables_min == 2
        assert cfg.num_variables_max == 2
        assert cfg.train_rows_min == 32
        assert cfg.train_rows_max == 32
        assert cfg.test_rows_min == 16
        assert cfg.test_rows_max == 16
        assert cfg.forecast_horizons == (1, 3, 6, 12)
        assert cfg.explicit_lags == (0, 1, 2, 5, 10)
        assert cfg.num_kernels == 3
        assert cfg.add_mask_channels is True
        assert cfg.missing_mode == "off"
        assert cfg.mechanism_type == "linear_var"
        assert cfg.noise_family == "normal"
        assert cfg.kernel_family == "exp_decay"


def test_temporal_profiles_use_shorter_common_budget(priors_modules) -> None:
    profiles_mod = priors_modules["research_profiles"]
    for name in profiles_mod.TEMPORAL_ABLATION_PROFILES:
        profile = profiles_mod.get_research_profile(name)
        assert profile.training_budget.epochs == 12
        assert profile.training_budget.steps == 400


def test_non_baseline_profiles_validate_on_raw_target_cfg(priors_modules) -> None:
    profiles_mod = priors_modules["research_profiles"]
    target_cfg = profiles_mod.target_cfg()

    for name in (
        "medium32k_live_guardrails",
        "medium32k_live_batch_homogeneous",
        "medium32k_live_mode_ladder",
        "medium32k_live_mixture",
    ):
        profile = profiles_mod.get_research_profile(name)
        assert profile.val_source.cfg == target_cfg


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
