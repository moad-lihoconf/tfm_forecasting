#!/usr/bin/env python3
"""Validate DynSCM generation invariants directly on sampled tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tfmplayground.priors.dynscm import sample_dynscm_variant_cfg
from tfmplayground.priors.dynscm.features import (
    _concat_feature_blocks,
    _extract_kernel_features,
    _extract_lag_features,
    _impute_with_train_means,
    _sample_kernels,
    build_forecasting_table,
    sample_origins_and_horizons,
)
from tfmplayground.priors.dynscm.graph import sample_regime_graphs
from tfmplayground.priors.dynscm.mechanisms import sample_regime_mechanisms
from tfmplayground.priors.dynscm.missingness import sample_observation_mask
from tfmplayground.priors.dynscm.parallel import (
    _SEED_MAX,
    compute_feasible_row_pairs,
    fit_feature_budget,
    prioritize_feature_blocks,
    sample_dataset_dimensions,
)
from tfmplayground.priors.dynscm.simulate import simulate_dynscm_series
from tfmplayground.priors.main import _DYNSCM_PROFILES, _load_dynscm_config


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dynscm-profile",
        type=str,
        default="rich_t4_96x128",
        choices=sorted(_DYNSCM_PROFILES.keys()),
    )
    parser.add_argument("--dynscm-config-json", type=str, default=None)
    parser.add_argument("--dynscm-override", action="append", default=[])
    parser.add_argument("--num-samples", type=int, default=32)
    parser.add_argument("--row-budget", type=int, default=None)
    parser.add_argument("--feature-budget", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out", type=str, default=None)
    return parser


def _resolve_dynscm_profile(
    *,
    dynscm_profile: str,
    dynscm_config_json: str | None,
    dynscm_overrides: list[str],
) -> tuple[Any, dict[str, object]]:
    profile = _DYNSCM_PROFILES[dynscm_profile]
    profile_cli_defaults = dict(profile.get("cli_defaults", {}))
    resolved_overrides = list(profile.get("dynscm_overrides", [])) + list(
        dynscm_overrides
    )
    cfg = _load_dynscm_config(dynscm_config_json, resolved_overrides)
    return cfg, profile_cli_defaults


def run_invariant_audit(
    *,
    dynscm_profile: str = "rich_t4_96x128",
    dynscm_config_json: str | None = None,
    dynscm_overrides: list[str] | None = None,
    num_samples: int = 32,
    row_budget: int | None = None,
    feature_budget: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    dynscm_overrides = list(dynscm_overrides or [])
    cfg, profile_cli_defaults = _resolve_dynscm_profile(
        dynscm_profile=dynscm_profile,
        dynscm_config_json=dynscm_config_json,
        dynscm_overrides=dynscm_overrides,
    )

    resolved_row_budget = int(
        row_budget
        if row_budget is not None
        else profile_cli_defaults.get("max_seq_len", 96)
    )
    resolved_feature_budget = int(
        feature_budget
        if feature_budget is not None
        else profile_cli_defaults.get("max_features", 128)
    )
    feasible_pairs = compute_feasible_row_pairs(
        cfg=cfg,
        num_datapoints_max=resolved_row_budget,
    )
    rng = np.random.default_rng(seed)

    label_mismatch_count = 0
    chronology_violation_count = 0
    pre_budget_mismatch_count = 0
    feature_block_order_violation_count = 0
    budget_slice_violation_count = 0
    label_visibility_violation_count = 0
    imputation_violation_count = 0

    observed_num_vars: list[int] = []
    observed_num_steps: list[int] = []
    observed_n_train: list[int] = []
    observed_n_test: list[int] = []
    observed_horizons: list[int] = []

    for _ in range(num_samples):
        sample_seed = int(rng.integers(0, _SEED_MAX, dtype=np.int64))
        sample_rng = cfg.make_rng(sample_seed)
        pair_idx = int(sample_rng.integers(0, feasible_pairs.shape[0]))
        n_train = int(feasible_pairs[pair_idx, 0])
        n_test = int(feasible_pairs[pair_idx, 1])

        variant_cfg, _variant_metadata = sample_dynscm_variant_cfg(cfg, sample_rng)
        num_vars, num_steps, y_idx = sample_dataset_dimensions(variant_cfg, sample_rng)
        observed_num_vars.append(int(num_vars))
        observed_num_steps.append(int(num_steps))
        observed_n_train.append(int(n_train))
        observed_n_test.append(int(n_test))

        per_sample_seeds = sample_rng.integers(0, _SEED_MAX, size=(6,), dtype=np.int64)
        graph_sample = sample_regime_graphs(
            variant_cfg,
            num_vars=num_vars,
            seed=int(per_sample_seeds[0]),
        )
        mechanism_sample = sample_regime_mechanisms(
            variant_cfg,
            graph_sample,
            seed=int(per_sample_seeds[1]),
        )
        simulation_sample = simulate_dynscm_series(
            variant_cfg,
            graph_sample,
            mechanism_sample,
            num_steps=num_steps,
            seed=int(per_sample_seeds[2]),
        )
        series = simulation_sample.series[None, :, :].astype(np.float64, copy=False)
        y_index = np.array([y_idx], dtype=np.int64)

        t_idx, h_idx = sample_origins_and_horizons(
            variant_cfg,
            batch_size=1,
            num_steps=num_steps,
            n_train=n_train,
            n_test=n_test,
            seed=int(per_sample_seeds[3]),
        )
        observed_horizons.extend(int(v) for v in h_idx.reshape(-1).tolist())
        obs_mask = sample_observation_mask(
            variant_cfg,
            series,
            seed=int(per_sample_seeds[4]),
            label_times=t_idx + h_idx,
            label_var_indices=y_index,
        )
        x_raw, y_raw, metadata = build_forecasting_table(
            variant_cfg,
            series,
            y_index,
            n_train=n_train,
            n_test=n_test,
            t_idx=t_idx,
            h_idx=h_idx,
            obs_mask=obs_mask,
            seed=int(per_sample_seeds[5]),
        )

        expected_y = series[0, (t_idx + h_idx)[0], y_idx]
        if not np.allclose(y_raw[0], expected_y, atol=1e-10, rtol=1e-10):
            label_mismatch_count += 1

        required_lag = max(variant_cfg.max_feature_lag, max(variant_cfg.explicit_lags))
        train_max = int(np.max(t_idx[0, :n_train]))
        test_min = int(np.min(t_idx[0, n_train:]))
        chronology_ok = (
            np.all(t_idx >= required_lag)
            and np.all((t_idx + h_idx) < num_steps)
            and train_max < test_min
        )
        if not chronology_ok:
            chronology_violation_count += 1

        feature_slices = metadata["feature_slices"]
        prioritized = prioritize_feature_blocks(x_raw, feature_slices=feature_slices)
        deterministic_block = x_raw[:, :, slice(*feature_slices["deterministic"])]
        lag_block = x_raw[:, :, slice(*feature_slices["lags_value"])]
        kernel_block = x_raw[:, :, slice(*feature_slices["kern_value"])]
        mask_block = (
            x_raw[:, :, slice(*feature_slices["data_mask"])]
            if "data_mask" in feature_slices
            else np.zeros((x_raw.shape[0], x_raw.shape[1], 0), dtype=x_raw.dtype)
        )
        ordered = np.concatenate(
            [deterministic_block, lag_block, kernel_block, mask_block],
            axis=2,
        )
        if not np.array_equal(prioritized, ordered):
            feature_block_order_violation_count += 1

        if int(prioritized.shape[2]) != int(ordered.shape[2]):
            pre_budget_mismatch_count += 1

        budgeted = fit_feature_budget(prioritized, num_features=resolved_feature_budget)
        if prioritized.shape[2] >= resolved_feature_budget:
            expected_budgeted = prioritized[:, :, :resolved_feature_budget]
        else:
            expected_budgeted = np.pad(
                prioritized,
                ((0, 0), (0, 0), (0, resolved_feature_budget - prioritized.shape[2])),
                mode="constant",
            )
        if not np.array_equal(budgeted, expected_budgeted):
            budget_slice_violation_count += 1

        if not np.all(obs_mask[0, (t_idx + h_idx)[0], y_idx]):
            label_visibility_violation_count += 1

        replay_rng = variant_cfg.make_rng(int(per_sample_seeds[5]))
        lag_values, lag_observed = _extract_lag_features(
            values=series,
            observed_mask=obs_mask,
            t_idx=t_idx,
            lags=np.asarray(variant_cfg.explicit_lags, dtype=np.int64),
        )
        kernels = _sample_kernels(cfg=variant_cfg, rng=replay_rng)
        kernel_values, kernel_observed = _extract_kernel_features(
            values=series,
            observed_mask=obs_mask,
            t_idx=t_idx,
            kernels=kernels,
        )
        data_values = _concat_feature_blocks([lag_values, kernel_values])
        data_observed = _concat_feature_blocks([lag_observed, kernel_observed]).astype(
            bool
        )
        imputed_data = _impute_with_train_means(
            values=data_values,
            observed=data_observed,
            n_train=n_train,
        )
        x_data_block = x_raw[:, :, slice(*feature_slices["data_values"])]
        if not np.allclose(imputed_data, x_data_block, atol=1e-10, rtol=1e-10):
            imputation_violation_count += 1

    observed_num_vars_arr = np.asarray(observed_num_vars, dtype=np.int64)
    observed_num_steps_arr = np.asarray(observed_num_steps, dtype=np.int64)
    observed_n_train_arr = np.asarray(observed_n_train, dtype=np.int64)
    observed_n_test_arr = np.asarray(observed_n_test, dtype=np.int64)
    observed_horizons_arr = np.asarray(observed_horizons, dtype=np.int64)

    expected_num_vars_support = set(
        range(cfg.num_variables_min, cfg.num_variables_max + 1)
    )
    expected_n_train_support = set(range(cfg.train_rows_min, cfg.train_rows_max + 1))
    expected_n_test_support = set(range(cfg.test_rows_min, cfg.test_rows_max + 1))
    expected_horizon_support = {int(v) for v in cfg.forecast_horizons}
    observed_horizon_support = {
        int(v) for v in np.unique(observed_horizons_arr).tolist()
    }

    sampling_support_ok = (
        set(np.unique(observed_num_vars_arr).tolist()).issubset(
            expected_num_vars_support
        )
        and set(np.unique(observed_n_train_arr).tolist()).issubset(
            expected_n_train_support
        )
        and set(np.unique(observed_n_test_arr).tolist()).issubset(
            expected_n_test_support
        )
        and observed_horizon_support == expected_horizon_support
        and int(np.min(observed_num_steps_arr)) >= int(cfg.series_length_min)
        and int(np.max(observed_num_steps_arr)) <= int(cfg.series_length_max)
    )

    invariants = {
        "label_correctness": {
            "status": "pass" if label_mismatch_count == 0 else "fail",
            "violations": int(label_mismatch_count),
        },
        "chronology_correctness": {
            "status": "pass" if chronology_violation_count == 0 else "fail",
            "violations": int(chronology_violation_count),
        },
        "feature_budget_semantics": {
            "status": (
                "pass"
                if pre_budget_mismatch_count == 0 and budget_slice_violation_count == 0
                else "fail"
            ),
            "pre_budget_count_violations": int(pre_budget_mismatch_count),
            "budget_slice_violations": int(budget_slice_violation_count),
        },
        "feature_block_ordering": {
            "status": "pass" if feature_block_order_violation_count == 0 else "fail",
            "violations": int(feature_block_order_violation_count),
        },
        "missingness_handling": {
            "status": (
                "pass"
                if label_visibility_violation_count == 0
                and imputation_violation_count == 0
                else "fail"
            ),
            "label_visibility_violations": int(label_visibility_violation_count),
            "train_only_imputation_violations": int(imputation_violation_count),
        },
        "sampling_support": {
            "status": "pass" if sampling_support_ok else "fail",
            "num_vars_support": sorted(
                int(v) for v in np.unique(observed_num_vars_arr).tolist()
            ),
            "num_steps_min": int(np.min(observed_num_steps_arr)),
            "num_steps_max": int(np.max(observed_num_steps_arr)),
            "n_train_support": sorted(
                int(v) for v in np.unique(observed_n_train_arr).tolist()
            ),
            "n_test_support": sorted(
                int(v) for v in np.unique(observed_n_test_arr).tolist()
            ),
            "horizon_support": sorted(int(v) for v in observed_horizon_support),
            "missing_horizons": sorted(
                int(v) for v in (expected_horizon_support - observed_horizon_support)
            ),
        },
    }

    return {
        "dynscm_profile": dynscm_profile,
        "dynscm_overrides": dynscm_overrides,
        "num_samples": int(num_samples),
        "row_budget": int(resolved_row_budget),
        "feature_budget": int(resolved_feature_budget),
        "seed": int(seed),
        "invariants": invariants,
        "status": (
            "pass"
            if all(v["status"] == "pass" for v in invariants.values())
            else "fail"
        ),
    }


def main() -> None:
    args = _build_parser().parse_args()
    payload = run_invariant_audit(
        dynscm_profile=args.dynscm_profile,
        dynscm_config_json=args.dynscm_config_json,
        dynscm_overrides=args.dynscm_override,
        num_samples=args.num_samples,
        row_budget=args.row_budget,
        feature_budget=args.feature_budget,
        seed=args.seed,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_out is not None:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
