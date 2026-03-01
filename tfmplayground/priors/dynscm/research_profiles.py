"""Named live DynSCM research profiles for medium-32k transition experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Literal, cast

from tfmplayground.priors.main import _DYNSCM_PROFILES, _load_dynscm_config

from .config import DynSCMConfig
from .research import (
    DynSCMBatchMode,
    DynSCMSampleFilterConfig,
    LinearPhaseSchedule,
)

DEFAULT_RESEARCH_WARM_START_CHECKPOINT = (
    "gs://tfm-forecasting-vertex-artifacts/tfm_forecasting/runs/"
    "dynscm-train-only-20260228-212512-medium-missing-16k/checkpoints/"
    "best_checkpoint.pth"
)
GUARDRAIL_MAX_SAMPLE_ATTEMPTS = 16
LIVE_MAX_SEQ_LEN = 48
LIVE_MAX_FEATURES = 64
COMMON_BATCH_SHARED_FIELDS = (
    "mechanism_type",
    "noise_family",
    "missing_mode",
    "kernel_family",
)
PHASE1_LIVE_PROFILES = (
    "medium32k_live_baseline",
    "medium32k_live_guardrails",
    "medium32k_live_batch_homogeneous",
    "medium32k_live_mode_ladder",
    "medium32k_live_mixture",
)
TEMPORAL_ABLATION_PROFILES = (
    "temporal_length_only_16k",
    "temporal_regimes_only_16k",
    "temporal_drift_only_16k",
    "temporal_regimes_plus_drift_16k",
    "temporal_length_plus_regimes_16k",
    "temporal_full_medium32k_reference",
)
BENCHMARK_CONTRACT_PROFILES = (
    "benchmark_contract_observed_easy",
    "benchmark_contract_observed_temporal",
)


@dataclass(frozen=True, slots=True)
class ResearchTrainingBudget:
    epochs: int = 20
    steps: int = 400
    batch_size: int = 16
    accumulate: int = 2
    lr: float = 5e-4
    weight_decay: float = 0.0
    dropout: float = 0.0
    amp: bool = True
    amp_dtype: Literal["float16", "bfloat16"] = "float16"
    eval_every_epochs: int = 1
    val_steps: int = 64
    early_stopping_metric: str = "val_loss"
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-5
    loss_weighting: Literal["per_target", "per_function"] = "per_target"
    debug_trace_first_n_batches: int = 200
    debug_trace_every_n_batches: int = 200


@dataclass(frozen=True, slots=True)
class SyntheticEvalSuiteSpec:
    name: str
    cfg: DynSCMConfig
    steps: int
    seed: int


@dataclass(frozen=True, slots=True)
class LiveSourceSpec:
    kind: Literal["single", "mode_ladder", "mixture"]
    cfg: DynSCMConfig | None
    sample_filter: DynSCMSampleFilterConfig | None = None
    max_sample_attempts_per_item: int = 1
    batch_shared_fields: tuple[str, ...] = ()
    modes: tuple[DynSCMBatchMode, ...] = ()
    schedule: LinearPhaseSchedule | None = None
    child_sources: tuple[tuple[str, DynSCMConfig], ...] = ()


@dataclass(frozen=True, slots=True)
class DynSCMLiveResearchProfile:
    name: str
    train_source: LiveSourceSpec
    val_source: LiveSourceSpec
    eval_suites: tuple[SyntheticEvalSuiteSpec, ...]
    training_budget: ResearchTrainingBudget
    warm_start_checkpoint: str
    train_seed: int
    val_seed: int
    max_seq_len: int = LIVE_MAX_SEQ_LEN
    max_features: int = LIVE_MAX_FEATURES


def _profile_cfg(profile_name: str) -> DynSCMConfig:
    profile = _DYNSCM_PROFILES[profile_name]
    raw_overrides = list(cast(list[str], profile.get("dynscm_overrides", [])))
    return _load_dynscm_config(None, raw_overrides)


@lru_cache(maxsize=1)
def stable_cfg() -> DynSCMConfig:
    return _profile_cfg("benchmark_aligned_medium_missing_16k")


@lru_cache(maxsize=1)
def target_cfg() -> DynSCMConfig:
    return _profile_cfg("benchmark_aligned_medium_32k")


@lru_cache(maxsize=1)
def full_cfg() -> DynSCMConfig:
    return _profile_cfg("benchmark_aligned_full_32k")


@lru_cache(maxsize=1)
def temporal_cfg() -> DynSCMConfig:
    return target_cfg().with_overrides(
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


@lru_cache(maxsize=1)
def temporal_length_only_cfg() -> DynSCMConfig:
    return stable_cfg().with_overrides(series_length_max=256)


@lru_cache(maxsize=1)
def temporal_regimes_only_cfg() -> DynSCMConfig:
    return stable_cfg().with_overrides(num_regimes=2, sticky_rho=0.95)


@lru_cache(maxsize=1)
def temporal_drift_only_cfg() -> DynSCMConfig:
    return stable_cfg().with_overrides(drift_std=0.01)


@lru_cache(maxsize=1)
def temporal_regimes_plus_drift_cfg() -> DynSCMConfig:
    return stable_cfg().with_overrides(num_regimes=2, sticky_rho=0.95, drift_std=0.01)


@lru_cache(maxsize=1)
def temporal_length_plus_regimes_cfg() -> DynSCMConfig:
    return stable_cfg().with_overrides(
        series_length_max=256,
        num_regimes=2,
        sticky_rho=0.95,
    )


@lru_cache(maxsize=1)
def temporal_full_medium32k_reference_cfg() -> DynSCMConfig:
    return target_cfg()


def _benchmark_contract_base_cfg() -> DynSCMConfig:
    return stable_cfg().with_overrides(
        num_variables_min=2,
        num_variables_max=2,
        train_rows_min=32,
        train_rows_max=32,
        test_rows_min=16,
        test_rows_max=16,
        forecast_horizons=(1, 3, 6, 12),
        explicit_lags=(0, 1, 2, 5, 10),
        num_kernels=3,
        max_feature_lag=32,
        add_mask_channels=True,
        use_contemp_edges=False,
        max_contemp_parents=0,
        max_lagged_parents=2,
        contemp_parent_rate=0.0,
        lagged_parent_rate=1.0,
        contemp_edge_add_prob=0.0,
        contemp_edge_del_prob=0.0,
        lagged_edge_add_prob=0.0,
        lagged_edge_del_prob=0.0,
        missing_mode="off",
        missing_mode_choices=None,
        missing_mode_probs=None,
        mechanism_type="linear_var",
        mechanism_type_choices=None,
        mechanism_type_probs=None,
        noise_family="normal",
        noise_family_choices=None,
        noise_family_probs=None,
        kernel_family="exp_decay",
        kernel_family_choices=None,
        kernel_family_probs=None,
        noise_scale_min=0.15,
        noise_scale_max=0.60,
    )


@lru_cache(maxsize=1)
def benchmark_contract_observed_easy_cfg() -> DynSCMConfig:
    return _benchmark_contract_base_cfg().with_overrides(
        series_length_min=128,
        series_length_max=128,
        num_regimes=1,
        sticky_rho=1.0,
        drift_std=0.0,
    )


@lru_cache(maxsize=1)
def benchmark_contract_observed_temporal_cfg() -> DynSCMConfig:
    return _benchmark_contract_base_cfg().with_overrides(
        series_length_min=128,
        series_length_max=256,
        num_regimes=2,
        sticky_rho=0.95,
        drift_std=0.01,
    )


def _guardrailed(cfg: DynSCMConfig) -> DynSCMConfig:
    return cfg.with_overrides(enable_spectral_rescale=True, spectral_radius_cap=0.90)


def _preserve_guardrail_strategy(
    source_cfg: DynSCMConfig | None,
    replacement_cfg: DynSCMConfig,
) -> DynSCMConfig:
    if source_cfg is None:
        return replacement_cfg
    if bool(getattr(source_cfg, "enable_spectral_rescale", False)):
        return replacement_cfg.with_overrides(
            enable_spectral_rescale=True,
            spectral_radius_cap=float(source_cfg.spectral_radius_cap),
        )
    return replacement_cfg


def _guardrail_filter() -> DynSCMSampleFilterConfig:
    return DynSCMSampleFilterConfig(
        reject_clipped=True,
        max_abs_value_cap=1000.0,
        min_train_target_std=1e-3,
    )


def _eval_suites() -> tuple[SyntheticEvalSuiteSpec, ...]:
    return (
        SyntheticEvalSuiteSpec(
            name="stable_eval",
            cfg=stable_cfg(),
            steps=64,
            seed=5402,
        ),
        SyntheticEvalSuiteSpec(
            name="temporal_eval",
            cfg=temporal_cfg(),
            steps=64,
            seed=6402,
        ),
        SyntheticEvalSuiteSpec(
            name="temporal_eval_hard",
            cfg=temporal_cfg(),
            steps=64,
            seed=6403,
        ),
        SyntheticEvalSuiteSpec(
            name="target_eval",
            cfg=target_cfg(),
            steps=64,
            seed=7402,
        ),
        SyntheticEvalSuiteSpec(
            name="full_eval",
            cfg=full_cfg(),
            steps=32,
            seed=8402,
        ),
    )


def _temporal_focus_eval_suites() -> tuple[SyntheticEvalSuiteSpec, ...]:
    return (
        SyntheticEvalSuiteSpec(
            name="stable_eval",
            cfg=stable_cfg(),
            steps=64,
            seed=5402,
        ),
        SyntheticEvalSuiteSpec(
            name="temporal_eval_hard",
            cfg=temporal_cfg(),
            steps=64,
            seed=6403,
        ),
        SyntheticEvalSuiteSpec(
            name="target_eval",
            cfg=target_cfg(),
            steps=64,
            seed=7402,
        ),
    )


def _single_source(
    cfg: DynSCMConfig,
    *,
    batch_shared_fields: tuple[str, ...] = (),
    sample_filter: DynSCMSampleFilterConfig | None = None,
    max_sample_attempts_per_item: int = 1,
) -> LiveSourceSpec:
    return LiveSourceSpec(
        kind="single",
        cfg=cfg,
        sample_filter=sample_filter,
        max_sample_attempts_per_item=max_sample_attempts_per_item,
        batch_shared_fields=batch_shared_fields,
    )


def _mode_ladder_modes() -> tuple[DynSCMBatchMode, ...]:
    easy_graph_overrides = {
        "num_regimes": 1,
        "sticky_rho": 1.0,
        "shared_order": True,
        "share_base_graph": True,
        "drift_std": 0.0,
        "series_length_min": 128,
        "series_length_max": 128,
        "lagged_edge_add_prob": 0.0,
        "lagged_edge_del_prob": 0.0,
        "mechanism_type": "linear_var",
        "mechanism_type_choices": None,
        "mechanism_type_probs": None,
        "noise_family": "normal",
        "noise_family_choices": None,
        "noise_family_probs": None,
        "missing_mode": "off",
        "missing_mode_choices": None,
        "missing_mode_probs": None,
        "kernel_family": "exp_decay",
        "kernel_family_choices": None,
        "kernel_family_probs": None,
    }
    easy_non_noise_overrides = {
        **easy_graph_overrides,
        "noise_family": "normal",
        "noise_family_choices": None,
        "noise_family_probs": None,
    }
    return (
        DynSCMBatchMode(
            name="graph_only",
            cfg_overrides={
                "num_regimes": 1,
                "sticky_rho": 1.0,
                "shared_order": True,
                "share_base_graph": True,
                "drift_std": 0.0,
                "series_length_min": 128,
                "series_length_max": 128,
                "noise_family": "normal",
                "noise_family_choices": None,
                "noise_family_probs": None,
                "missing_mode": "off",
                "missing_mode_choices": None,
                "missing_mode_probs": None,
                "kernel_family": "exp_decay",
                "kernel_family_choices": None,
                "kernel_family_probs": None,
            },
        ),
        DynSCMBatchMode(
            name="noise_only",
            cfg_overrides={
                **easy_graph_overrides,
                "noise_family": "student_t",
                "noise_family_choices": ("normal", "student_t"),
                "noise_family_probs": (0.8, 0.2),
            },
        ),
        DynSCMBatchMode(
            name="missing_only",
            cfg_overrides={
                **easy_non_noise_overrides,
                "missing_mode": "mix",
                "missing_mode_choices": ("off", "mcar"),
                "missing_mode_probs": (0.8, 0.2),
            },
        ),
        DynSCMBatchMode(
            name="temporal_only",
            cfg_overrides={
                "lagged_edge_add_prob": 0.0,
                "lagged_edge_del_prob": 0.0,
                "mechanism_type": "linear_var",
                "mechanism_type_choices": None,
                "mechanism_type_probs": None,
                "noise_family": "normal",
                "noise_family_choices": None,
                "noise_family_probs": None,
                "missing_mode": "off",
                "missing_mode_choices": None,
                "missing_mode_probs": None,
                "kernel_family": "exp_decay",
                "kernel_family_choices": None,
                "kernel_family_probs": None,
            },
        ),
        DynSCMBatchMode(name="full", cfg_overrides={}),
    )


def _mode_ladder_schedule() -> LinearPhaseSchedule:
    return LinearPhaseSchedule(
        [
            (0.5, (0.25, 0.25, 0.25, 0.25, 0.0), None),
            (0.8, (0.25, 0.25, 0.25, 0.25, 0.0), (0.175, 0.175, 0.175, 0.175, 0.30)),
            (1.0, (0.175, 0.175, 0.175, 0.175, 0.30), (0.10, 0.10, 0.10, 0.10, 0.60)),
        ]
    )


def _mixture_schedule() -> LinearPhaseSchedule:
    return LinearPhaseSchedule(
        [
            (0.3, (1.0, 0.0, 0.0), None),
            (0.7, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            (1.0, (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ]
    )


@lru_cache(maxsize=1)
def research_profiles() -> dict[str, DynSCMLiveResearchProfile]:
    budget = ResearchTrainingBudget()
    temporal_budget = ResearchTrainingBudget(epochs=12)
    eval_suites = _eval_suites()
    temporal_eval_suites = _temporal_focus_eval_suites()
    target_guardrailed = _guardrailed(target_cfg())
    stable_guardrailed = _guardrailed(stable_cfg())
    temporal_guardrailed = _guardrailed(temporal_cfg())
    common_filter = _guardrail_filter()

    baseline = DynSCMLiveResearchProfile(
        name="medium32k_live_baseline",
        train_source=_single_source(target_cfg()),
        val_source=_single_source(target_cfg()),
        eval_suites=eval_suites,
        training_budget=budget,
        warm_start_checkpoint=DEFAULT_RESEARCH_WARM_START_CHECKPOINT,
        train_seed=2402,
        val_seed=3402,
    )
    guardrails = DynSCMLiveResearchProfile(
        name="medium32k_live_guardrails",
        train_source=_single_source(
            target_guardrailed,
            sample_filter=common_filter,
            max_sample_attempts_per_item=GUARDRAIL_MAX_SAMPLE_ATTEMPTS,
        ),
        val_source=_single_source(target_cfg()),
        eval_suites=eval_suites,
        training_budget=budget,
        warm_start_checkpoint=DEFAULT_RESEARCH_WARM_START_CHECKPOINT,
        train_seed=2402,
        val_seed=3402,
    )
    batch_homogeneous = DynSCMLiveResearchProfile(
        name="medium32k_live_batch_homogeneous",
        train_source=_single_source(
            target_guardrailed,
            batch_shared_fields=COMMON_BATCH_SHARED_FIELDS,
            sample_filter=common_filter,
            max_sample_attempts_per_item=GUARDRAIL_MAX_SAMPLE_ATTEMPTS,
        ),
        val_source=_single_source(target_cfg()),
        eval_suites=eval_suites,
        training_budget=budget,
        warm_start_checkpoint=DEFAULT_RESEARCH_WARM_START_CHECKPOINT,
        train_seed=2402,
        val_seed=3402,
    )
    mode_ladder = DynSCMLiveResearchProfile(
        name="medium32k_live_mode_ladder",
        train_source=LiveSourceSpec(
            kind="mode_ladder",
            cfg=target_guardrailed,
            sample_filter=common_filter,
            max_sample_attempts_per_item=GUARDRAIL_MAX_SAMPLE_ATTEMPTS,
            batch_shared_fields=COMMON_BATCH_SHARED_FIELDS,
            modes=_mode_ladder_modes(),
            schedule=_mode_ladder_schedule(),
        ),
        val_source=_single_source(target_cfg()),
        eval_suites=eval_suites,
        training_budget=budget,
        warm_start_checkpoint=DEFAULT_RESEARCH_WARM_START_CHECKPOINT,
        train_seed=2402,
        val_seed=3402,
    )
    mixture = DynSCMLiveResearchProfile(
        name="medium32k_live_mixture",
        train_source=LiveSourceSpec(
            kind="mixture",
            cfg=None,
            sample_filter=common_filter,
            max_sample_attempts_per_item=GUARDRAIL_MAX_SAMPLE_ATTEMPTS,
            batch_shared_fields=COMMON_BATCH_SHARED_FIELDS,
            schedule=_mixture_schedule(),
            child_sources=(
                ("stable_cfg", stable_guardrailed),
                ("temporal_cfg", temporal_guardrailed),
                ("target_cfg", target_guardrailed),
            ),
        ),
        val_source=_single_source(target_cfg()),
        eval_suites=eval_suites,
        training_budget=budget,
        warm_start_checkpoint=DEFAULT_RESEARCH_WARM_START_CHECKPOINT,
        train_seed=2402,
        val_seed=3402,
    )
    temporal_length_only = DynSCMLiveResearchProfile(
        name="temporal_length_only_16k",
        train_source=_single_source(temporal_length_only_cfg()),
        val_source=_single_source(target_cfg()),
        eval_suites=temporal_eval_suites,
        training_budget=temporal_budget,
        warm_start_checkpoint=DEFAULT_RESEARCH_WARM_START_CHECKPOINT,
        train_seed=2502,
        val_seed=3502,
    )
    temporal_regimes_only = DynSCMLiveResearchProfile(
        name="temporal_regimes_only_16k",
        train_source=_single_source(temporal_regimes_only_cfg()),
        val_source=_single_source(target_cfg()),
        eval_suites=temporal_eval_suites,
        training_budget=temporal_budget,
        warm_start_checkpoint=DEFAULT_RESEARCH_WARM_START_CHECKPOINT,
        train_seed=2503,
        val_seed=3503,
    )
    temporal_drift_only = DynSCMLiveResearchProfile(
        name="temporal_drift_only_16k",
        train_source=_single_source(temporal_drift_only_cfg()),
        val_source=_single_source(target_cfg()),
        eval_suites=temporal_eval_suites,
        training_budget=temporal_budget,
        warm_start_checkpoint=DEFAULT_RESEARCH_WARM_START_CHECKPOINT,
        train_seed=2504,
        val_seed=3504,
    )
    temporal_regimes_plus_drift = DynSCMLiveResearchProfile(
        name="temporal_regimes_plus_drift_16k",
        train_source=_single_source(temporal_regimes_plus_drift_cfg()),
        val_source=_single_source(target_cfg()),
        eval_suites=temporal_eval_suites,
        training_budget=temporal_budget,
        warm_start_checkpoint=DEFAULT_RESEARCH_WARM_START_CHECKPOINT,
        train_seed=2505,
        val_seed=3505,
    )
    temporal_length_plus_regimes = DynSCMLiveResearchProfile(
        name="temporal_length_plus_regimes_16k",
        train_source=_single_source(temporal_length_plus_regimes_cfg()),
        val_source=_single_source(target_cfg()),
        eval_suites=temporal_eval_suites,
        training_budget=temporal_budget,
        warm_start_checkpoint=DEFAULT_RESEARCH_WARM_START_CHECKPOINT,
        train_seed=2506,
        val_seed=3506,
    )
    temporal_full_reference = DynSCMLiveResearchProfile(
        name="temporal_full_medium32k_reference",
        train_source=_single_source(temporal_full_medium32k_reference_cfg()),
        val_source=_single_source(target_cfg()),
        eval_suites=temporal_eval_suites,
        training_budget=temporal_budget,
        warm_start_checkpoint=DEFAULT_RESEARCH_WARM_START_CHECKPOINT,
        train_seed=2507,
        val_seed=3507,
    )
    benchmark_contract_easy = DynSCMLiveResearchProfile(
        name="benchmark_contract_observed_easy",
        train_source=_single_source(benchmark_contract_observed_easy_cfg()),
        val_source=_single_source(benchmark_contract_observed_easy_cfg()),
        eval_suites=temporal_eval_suites,
        training_budget=budget,
        warm_start_checkpoint=DEFAULT_RESEARCH_WARM_START_CHECKPOINT,
        train_seed=2602,
        val_seed=3602,
    )
    benchmark_contract_temporal = DynSCMLiveResearchProfile(
        name="benchmark_contract_observed_temporal",
        train_source=_single_source(benchmark_contract_observed_temporal_cfg()),
        val_source=_single_source(benchmark_contract_observed_temporal_cfg()),
        eval_suites=temporal_eval_suites,
        training_budget=budget,
        warm_start_checkpoint=DEFAULT_RESEARCH_WARM_START_CHECKPOINT,
        train_seed=2603,
        val_seed=3603,
    )
    return {
        profile.name: profile
        for profile in (
            baseline,
            guardrails,
            batch_homogeneous,
            mode_ladder,
            mixture,
            temporal_length_only,
            temporal_regimes_only,
            temporal_drift_only,
            temporal_regimes_plus_drift,
            temporal_length_plus_regimes,
            temporal_full_reference,
            benchmark_contract_easy,
            benchmark_contract_temporal,
        )
    }


def get_research_profile(name: str) -> DynSCMLiveResearchProfile:
    try:
        return research_profiles()[name]
    except KeyError as exc:
        known = ", ".join(sorted(research_profiles()))
        raise KeyError(f"Unknown research profile {name!r}. Known: {known}.") from exc


def list_research_profiles() -> tuple[str, ...]:
    return tuple(research_profiles().keys())


def _promote_source_to_full(source: LiveSourceSpec) -> LiveSourceSpec:
    if source.kind == "single":
        return replace(
            source,
            cfg=_preserve_guardrail_strategy(source.cfg, full_cfg()),
        )
    if source.kind == "mode_ladder":
        return replace(
            source,
            cfg=_preserve_guardrail_strategy(source.cfg, full_cfg()),
        )
    if source.kind == "mixture":
        promoted_children: list[tuple[str, DynSCMConfig]] = []
        for child_name, child_cfg in source.child_sources:
            if child_name == "target_cfg":
                promoted_children.append(
                    (
                        "full_cfg",
                        _preserve_guardrail_strategy(child_cfg, full_cfg()),
                    )
                )
            else:
                promoted_children.append((child_name, child_cfg))
        return replace(source, child_sources=tuple(promoted_children))
    raise ValueError(f"Unsupported source kind for promotion: {source.kind!r}.")


def build_promotion_profile(name: str) -> DynSCMLiveResearchProfile:
    base_profile = get_research_profile(name)
    return replace(
        base_profile,
        name=f"{base_profile.name}_full_promotion",
        train_source=_promote_source_to_full(base_profile.train_source),
    )
