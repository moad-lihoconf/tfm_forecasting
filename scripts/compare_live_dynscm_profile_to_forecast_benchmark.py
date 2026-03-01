#!/usr/bin/env python3
"""Compare a live DynSCM research profile source against the forecast benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import torch

import pretrain_regression_dynscm_live as live_train
from scripts.compare_dynscm_prior_to_forecast_benchmark import (
    _deterministic_feature_width,
    _distribution,
    _feature_block_counts,
    _safe_stats,
    _support,
    build_markdown_report,
    build_mismatch_report,
    summarize_benchmark,
)
from tfmplayground.benchmarks.forecasting.config import ForecastBenchmarkConfig
from tfmplayground.priors.dynscm.research_profiles import (
    DynSCMLiveResearchProfile,
    LiveSourceSpec,
    get_research_profile,
)
from tfmplayground.utils import get_default_device

_MECHANISM_NAMES = ("linear_var", "linear_plus_residual")
_NOISE_NAMES = ("normal", "student_t")
_MISSING_NAMES = ("off", "mcar", "mar", "mnar_lite", "mix")
_KERNEL_NAMES = ("exp_decay", "power_law", "mix")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research_profile", type=str, required=True)
    parser.add_argument(
        "--source",
        type=str,
        default="train",
        choices=("train", "val"),
        help="which live source to summarize",
    )
    parser.add_argument("--sample_steps", type=int, default=64)
    parser.add_argument("--batchsize", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dynscm_workers", type=int, default=4)
    parser.add_argument("--dynscm_worker_blas_threads", type=int, default=1)
    parser.add_argument("--benchmark-config-json", type=str, default=None)
    parser.add_argument("--dataset-limit", type=int, default=None)
    parser.add_argument("--series-limit", type=int, default=None)
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--markdown-out", type=str, default=None)
    parser.add_argument("--prior-audit-json", type=str, default=None)
    return parser


def _source_for_name(
    profile: DynSCMLiveResearchProfile,
    source_name: str,
) -> LiveSourceSpec:
    if source_name == "train":
        return profile.train_source
    if source_name == "val":
        return profile.val_source
    raise ValueError(f"Unsupported source name: {source_name!r}.")


def _cfg_reference_for_source(source: LiveSourceSpec):
    if source.cfg is not None:
        return source.cfg
    if source.child_sources:
        return source.child_sources[0][1]
    raise ValueError("Could not resolve a reference config for source.")


def _tensor_batch_item(
    batch: dict[str, torch.Tensor | int],
    key: str,
) -> torch.Tensor:
    value = batch[key]
    if not torch.is_tensor(value):
        raise TypeError(f"Expected tensor batch field {key!r}, got {type(value)}.")
    return cast(torch.Tensor, value)


def _named_distribution(values: list[int], names: tuple[str, ...]) -> dict[str, float]:
    if not values:
        return {}
    unique, counts = np.unique(np.asarray(values, dtype=np.int64), return_counts=True)
    probs = counts.astype(np.float64) / float(np.sum(counts))
    return {
        names[int(index)]: float(probability)
        for index, probability in zip(unique.tolist(), probs.tolist(), strict=True)
        if 0 <= int(index) < len(names)
    }


def summarize_live_source(
    *,
    profile: DynSCMLiveResearchProfile,
    source_name: str,
    sample_steps: int,
    batch_size: int,
    device: torch.device,
    workers: int,
    worker_blas_threads: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    source = _source_for_name(profile, source_name)
    cfg_ref = _cfg_reference_for_source(source)
    loader = live_train._build_prior_loader(
        source=source,
        num_steps=sample_steps,
        batch_size=batch_size,
        num_datapoints_max=profile.max_seq_len,
        num_features=profile.max_features,
        device=device,
        seed=profile.train_seed if source_name == "train" else profile.val_seed,
        workers=workers,
        worker_blas_threads=worker_blas_threads,
        total_train_batches=max(1, sample_steps),
    )

    context_rows: list[int] = []
    test_rows: list[int] = []
    target_rows: list[int] = []
    num_variables: list[int] = []
    active_feature_count: list[int] = []
    feature_count_before_padding: list[int] = []
    train_target_std: list[float] = []
    test_target_std: list[float] = []
    mechanism_ids: list[int] = []
    noise_ids: list[int] = []
    missing_ids: list[int] = []
    kernel_ids: list[int] = []
    low_variance_count = 0
    processed_batches = 0

    try:
        for batch in loader:
            processed_batches += 1
            if processed_batches == 1 or processed_batches % 8 == 0:
                print(
                    "[benchmark-compare] progress "
                    f"batch={processed_batches}/{sample_steps}",
                    flush=True,
                )
            x = _tensor_batch_item(batch, "x").detach().cpu().numpy()
            y = _tensor_batch_item(batch, "y").detach().cpu().numpy()
            sampled_n_train = (
                _tensor_batch_item(batch, "sampled_n_train").detach().cpu().numpy()
            )
            sampled_n_test = (
                _tensor_batch_item(batch, "sampled_n_test").detach().cpu().numpy()
            )
            sampled_num_vars = (
                _tensor_batch_item(batch, "sampled_num_vars").detach().cpu().numpy()
            )
            sampled_pre_budget = (
                _tensor_batch_item(
                    batch,
                    "sampled_pre_budget_feature_count",
                )
                .detach()
                .cpu()
                .numpy()
            )
            sampled_mech = (
                _tensor_batch_item(batch, "sampled_mechanism_type_id")
                .detach()
                .cpu()
                .numpy()
            )
            sampled_noise = (
                _tensor_batch_item(batch, "sampled_noise_family_id")
                .detach()
                .cpu()
                .numpy()
            )
            sampled_missing = (
                _tensor_batch_item(batch, "sampled_missing_mode_id")
                .detach()
                .cpu()
                .numpy()
            )
            sampled_kernel = (
                _tensor_batch_item(batch, "sampled_kernel_family_id")
                .detach()
                .cpu()
                .numpy()
            )

            for idx in range(x.shape[0]):
                n_train = int(sampled_n_train[idx])
                n_test = int(sampled_n_test[idx])
                num_rows = n_train + n_test
                x_i = x[idx, :num_rows]
                y_i = y[idx, :num_rows]
                context_rows.append(n_train)
                test_rows.append(n_test)
                target_rows.append(n_test)
                num_variables.append(int(sampled_num_vars[idx]))
                feature_count_before_padding.append(int(sampled_pre_budget[idx]))
                active_feature_count.append(
                    int(np.any(np.abs(x_i) > 1e-12, axis=0).sum())
                )

                train_std = float(np.std(y_i[:n_train], ddof=1 if n_train > 1 else 0))
                test_std = float(np.std(y_i[n_train:], ddof=1 if n_test > 1 else 0))
                train_target_std.append(train_std)
                test_target_std.append(test_std)
                if test_std < 1e-3:
                    low_variance_count += 1

                mechanism_ids.append(int(sampled_mech[idx]))
                noise_ids.append(int(sampled_noise[idx]))
                missing_ids.append(int(sampled_missing[idx]))
                kernel_ids.append(int(sampled_kernel[idx]))
    finally:
        loader.close()

    num_variables_arr = np.asarray(num_variables, dtype=np.int64)
    context_rows_arr = np.asarray(context_rows, dtype=np.int64)
    test_rows_arr = np.asarray(test_rows, dtype=np.int64)
    target_rows_arr = np.asarray(target_rows, dtype=np.int64)
    feature_count_arr = np.asarray(feature_count_before_padding, dtype=np.int64)
    active_feature_arr = np.asarray(active_feature_count, dtype=np.int64)
    train_target_std_arr = np.asarray(train_target_std, dtype=np.float64)
    test_target_std_arr = np.asarray(test_target_std, dtype=np.float64)
    horizons_arr = np.asarray(list(cfg_ref.forecast_horizons), dtype=np.int64)

    summary = {
        "source": "live_dynscm_profile",
        "research_profile": profile.name,
        "live_source": source_name,
        "inspected_tables": int(num_variables_arr.size),
        "num_variables": {
            "support": _support(num_variables_arr),
            "distribution": _distribution(num_variables_arr),
            "stats": _safe_stats(num_variables_arr.astype(np.float64)),
        },
        "context_rows": {
            "support": _support(context_rows_arr),
            "distribution": _distribution(context_rows_arr),
            "stats": _safe_stats(context_rows_arr.astype(np.float64)),
        },
        "test_rows": {
            "support": _support(test_rows_arr),
            "distribution": _distribution(test_rows_arr),
            "stats": _safe_stats(test_rows_arr.astype(np.float64)),
        },
        "horizons": {
            "support": _support(horizons_arr),
            "distribution": _distribution(horizons_arr),
        },
        "feature_count_before_padding": {
            "support": _support(feature_count_arr),
            "stats": _safe_stats(feature_count_arr.astype(np.float64)),
        },
        "active_feature_count": {
            "support": _support(active_feature_arr),
            "stats": _safe_stats(active_feature_arr.astype(np.float64)),
        },
        "lag_set": list(cfg_ref.explicit_lags),
        "num_kernels": int(cfg_ref.num_kernels),
        "mask_channels": bool(cfg_ref.add_mask_channels),
        "target_row_count": {
            "support": _support(target_rows_arr),
            "distribution": _distribution(target_rows_arr),
            "stats": _safe_stats(target_rows_arr.astype(np.float64)),
        },
        "train_target_std": _safe_stats(train_target_std_arr),
        "test_target_std": _safe_stats(test_target_std_arr),
        "low_variance_target_fraction": (
            float(low_variance_count / max(1, int(num_variables_arr.size)))
        ),
        "missing_mode_distribution": _named_distribution(missing_ids, _MISSING_NAMES),
        "family_distributions": {
            "mechanism_type": _named_distribution(mechanism_ids, _MECHANISM_NAMES),
            "noise_family": _named_distribution(noise_ids, _NOISE_NAMES),
            "missing_mode": _named_distribution(missing_ids, _MISSING_NAMES),
            "kernel_family": _named_distribution(kernel_ids, _KERNEL_NAMES),
        },
        "feature_block_counts": _feature_block_counts(
            num_variables=num_variables_arr,
            explicit_lags=list(cfg_ref.explicit_lags),
            num_kernels=int(cfg_ref.num_kernels),
            add_mask_channels=bool(cfg_ref.add_mask_channels),
            deterministic_width=_deterministic_feature_width(cfg_ref.to_dict()),
        ),
    }
    live_samples = {
        "num_variables": num_variables_arr,
        "context_rows": context_rows_arr,
        "test_rows": test_rows_arr,
        "horizons": horizons_arr,
        "feature_count_before_padding": feature_count_arr,
        "active_feature_count": active_feature_arr,
        "target_row_count": target_rows_arr,
        "train_target_std": train_target_std_arr,
        "test_target_std": test_target_std_arr,
    }
    print(
        "[benchmark-compare] done "
        f"batches={processed_batches} tables={int(num_variables_arr.size)}",
        flush=True,
    )
    return summary, live_samples


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    profile = get_research_profile(args.research_profile)
    device = torch.device(args.device or get_default_device())
    sample_batch_size = (
        int(args.batchsize)
        if args.batchsize is not None
        else int(profile.training_budget.batch_size)
    )
    benchmark_cfg = (
        ForecastBenchmarkConfig.from_json(args.benchmark_config_json)
        if args.benchmark_config_json is not None
        else ForecastBenchmarkConfig()
    )

    live_summary, live_samples = summarize_live_source(
        profile=profile,
        source_name=args.source,
        sample_steps=int(args.sample_steps),
        batch_size=sample_batch_size,
        device=device,
        workers=int(args.dynscm_workers),
        worker_blas_threads=int(args.dynscm_worker_blas_threads),
    )
    benchmark_summary, benchmark_samples = summarize_benchmark(
        benchmark_cfg,
        dataset_limit=args.dataset_limit,
        series_limit=args.series_limit,
    )
    mismatches = build_mismatch_report(
        live_summary,
        live_samples,
        benchmark_summary,
        benchmark_samples,
    )
    payload = {
        "research_profile": profile.name,
        "source": args.source,
        "prior_summary": live_summary,
        "benchmark_summary": benchmark_summary,
        "mismatches": mismatches,
    }

    if args.prior_audit_json is not None:
        prior_audit_path = Path(args.prior_audit_json)
        prior_audit_path.parent.mkdir(parents=True, exist_ok=True)
        prior_audit_path.write_text(
            json.dumps(live_summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if args.json_out is not None:
        json_out_path = Path(args.json_out)
        json_out_path.parent.mkdir(parents=True, exist_ok=True)
        json_out_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if args.markdown_out is not None:
        markdown_path = Path(args.markdown_out)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(
            build_markdown_report(payload),
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
