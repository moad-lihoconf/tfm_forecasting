#!/usr/bin/env python3
"""Compare a DynSCM prior dump against the real forecasting benchmark protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np

from tfmplayground.benchmarks.forecasting.adapters import (
    build_forecast_table_from_series,
)
from tfmplayground.benchmarks.forecasting.config import ForecastBenchmarkConfig
from tfmplayground.benchmarks.forecasting.datasets import load_suite
from tfmplayground.benchmarks.forecasting.protocol import (
    generate_rolling_origin_indices,
)
from tfmplayground.priors.audit import audit_prior_dump


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priordump", type=str, required=True)
    parser.add_argument("--benchmark-config-json", type=str, default=None)
    parser.add_argument("--dataset-limit", type=int, default=None)
    parser.add_argument("--series-limit", type=int, default=None)
    parser.add_argument("--sample-limit", type=int, default=4096)
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument(
        "--allow-download",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--markdown-out", type=str, default=None)
    return parser


def _load_dump_metadata(priordump: str) -> dict[str, object]:
    with h5py.File(priordump, "r") as f:
        if "dump_metadata_json" not in f:
            return {}
        raw_metadata = f["dump_metadata_json"][()]
    metadata_text = (
        raw_metadata.decode("utf-8")
        if isinstance(raw_metadata, bytes)
        else str(raw_metadata)
    )
    try:
        payload = json.loads(metadata_text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _safe_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "min": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "max": float("nan"),
        }
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "max": float(np.max(values)),
    }


def _support(values: np.ndarray) -> list[int]:
    if values.size == 0:
        return []
    return [int(v) for v in np.unique(values).tolist()]


def _distribution(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    unique, counts = np.unique(values, return_counts=True)
    probs = counts.astype(np.float64) / float(np.sum(counts))
    return {
        str(int(value)): float(probability)
        for value, probability in zip(unique.tolist(), probs.tolist(), strict=True)
    }


def _empirical_wasserstein_1d(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    a_sorted = np.sort(a.astype(np.float64))
    b_sorted = np.sort(b.astype(np.float64))
    quantiles = np.linspace(
        0.0, 1.0, num=max(a_sorted.size, b_sorted.size), endpoint=True
    )
    a_interp = np.quantile(a_sorted, quantiles, method="linear")
    b_interp = np.quantile(b_sorted, quantiles, method="linear")
    return float(np.mean(np.abs(a_interp - b_interp)))


def _support_overlap(prior_support: list[int], benchmark_support: list[int]) -> float:
    prior_set = set(prior_support)
    benchmark_set = set(benchmark_support)
    if not prior_set and not benchmark_set:
        return 1.0
    union = prior_set | benchmark_set
    if not union:
        return 1.0
    return float(len(prior_set & benchmark_set) / len(union))


def _total_variation_distance(
    prior_distribution: dict[str, float],
    benchmark_distribution: dict[str, float],
) -> float:
    keys = set(prior_distribution) | set(benchmark_distribution)
    if not keys:
        return float("nan")
    return 0.5 * sum(
        abs(prior_distribution.get(key, 0.0) - benchmark_distribution.get(key, 0.0))
        for key in keys
    )


def _deterministic_feature_width(config_like: dict[str, Any]) -> int:
    return (
        int(bool(config_like.get("add_time_feature", True)))
        + int(bool(config_like.get("add_horizon_feature", True)))
        + int(bool(config_like.get("add_log_horizon", True)))
        + (2 if bool(config_like.get("add_seasonality", True)) else 0)
    )


def _feature_block_counts(
    *,
    num_variables: np.ndarray,
    explicit_lags: list[int],
    num_kernels: int,
    add_mask_channels: bool,
    deterministic_width: int,
) -> dict[str, dict[str, float]]:
    if num_variables.size == 0:
        return {}
    lag_width = num_variables * len(explicit_lags)
    kernel_width = num_variables * int(num_kernels)
    data_width = lag_width + kernel_width
    mask_width = data_width if add_mask_channels else np.zeros_like(data_width)
    det_width = np.full_like(num_variables, deterministic_width)
    total_width = data_width + mask_width + det_width
    return {
        "deterministic": _safe_stats(det_width.astype(np.float64)),
        "lag_values": _safe_stats(lag_width.astype(np.float64)),
        "kernel_values": _safe_stats(kernel_width.astype(np.float64)),
        "data_mask": _safe_stats(mask_width.astype(np.float64)),
        "total_pre_budget_theoretical": _safe_stats(total_width.astype(np.float64)),
    }


def summarize_prior_dump(
    priordump: str,
    *,
    sample_limit: int | None = 4096,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    audit = cast(dict[str, Any], audit_prior_dump(priordump, sample_limit=sample_limit))
    metadata = _load_dump_metadata(priordump)
    dynscm_cfg = metadata.get("dynscm_config")
    dynscm_cfg = dynscm_cfg if isinstance(dynscm_cfg, dict) else {}
    explicit_lags = [int(v) for v in dynscm_cfg.get("explicit_lags", [])]
    num_kernels = int(dynscm_cfg.get("num_kernels", 0))
    add_mask_channels = bool(dynscm_cfg.get("add_mask_channels", False))
    deterministic_width = _deterministic_feature_width(dynscm_cfg)

    with h5py.File(priordump, "r") as f:
        num_rows = int(f["X"].shape[0])
        inspected = (
            num_rows if sample_limit is None else min(num_rows, int(sample_limit))
        )
        row_ids = np.arange(inspected, dtype=np.int64)
        x = np.asarray(f["X"][row_ids], dtype=np.float32)
        y = np.asarray(f["y"][row_ids], dtype=np.float32)
        num_datapoints = np.asarray(f["num_datapoints"][row_ids], dtype=np.int64)
        single_eval_pos = np.asarray(f["single_eval_pos"][row_ids], dtype=np.int64)
        num_vars = (
            np.asarray(f["sampled_num_vars"][row_ids], dtype=np.int64)
            if "sampled_num_vars" in f
            else np.asarray([], dtype=np.int64)
        )
        n_train = (
            np.asarray(f["sampled_n_train"][row_ids], dtype=np.int64)
            if "sampled_n_train" in f
            else single_eval_pos.copy()
        )
        n_test = (
            np.asarray(f["sampled_n_test"][row_ids], dtype=np.int64)
            if "sampled_n_test" in f
            else np.clip(num_datapoints - single_eval_pos, 0, None)
        )
        pre_budget = (
            np.asarray(f["sampled_pre_budget_feature_count"][row_ids], dtype=np.int64)
            if "sampled_pre_budget_feature_count" in f
            else np.asarray([], dtype=np.int64)
        )

    active_features = np.any(np.abs(x) > 1e-12, axis=1).sum(axis=1).astype(np.int64)
    target_rows = np.clip(num_datapoints - single_eval_pos, 0, None)
    train_stds: list[float] = []
    test_stds: list[float] = []
    for idx in range(inspected):
        sep = int(single_eval_pos[idx])
        nd = int(num_datapoints[idx])
        train_target = y[idx, :sep]
        test_target = y[idx, sep:nd]
        if train_target.size > 0:
            ddof = 1 if train_target.size > 1 else 0
            train_stds.append(float(np.std(train_target, ddof=ddof)))
        if test_target.size > 0:
            ddof = 1 if test_target.size > 1 else 0
            test_stds.append(float(np.std(test_target, ddof=ddof)))

    prior_samples = {
        "num_variables": num_vars,
        "context_rows": n_train,
        "test_rows": n_test,
        "feature_count_before_padding": pre_budget,
        "active_feature_count": active_features,
        "target_row_count": target_rows,
        "train_target_std": np.asarray(train_stds, dtype=np.float64),
        "test_target_std": np.asarray(test_stds, dtype=np.float64),
        "horizons": np.asarray(
            [int(v) for v in dynscm_cfg.get("forecast_horizons", [])],
            dtype=np.int64,
        ),
    }

    summary = {
        "source": "prior_dump",
        "priordump": priordump,
        "num_tables": int(num_rows),
        "inspected_tables": int(inspected),
        "num_variables": {
            "support": _support(prior_samples["num_variables"]),
            "distribution": _distribution(prior_samples["num_variables"]),
            "stats": _safe_stats(prior_samples["num_variables"].astype(np.float64)),
        },
        "context_rows": {
            "support": _support(prior_samples["context_rows"]),
            "distribution": _distribution(prior_samples["context_rows"]),
            "stats": _safe_stats(prior_samples["context_rows"].astype(np.float64)),
        },
        "test_rows": {
            "support": _support(prior_samples["test_rows"]),
            "distribution": _distribution(prior_samples["test_rows"]),
            "stats": _safe_stats(prior_samples["test_rows"].astype(np.float64)),
        },
        "horizons": {
            "support": _support(prior_samples["horizons"]),
            "distribution": _distribution(prior_samples["horizons"]),
        },
        "feature_count_before_padding": {
            "support": _support(prior_samples["feature_count_before_padding"]),
            "stats": _safe_stats(
                prior_samples["feature_count_before_padding"].astype(np.float64)
            ),
        },
        "active_feature_count": {
            "support": _support(prior_samples["active_feature_count"]),
            "stats": _safe_stats(
                prior_samples["active_feature_count"].astype(np.float64)
            ),
        },
        "lag_set": explicit_lags,
        "num_kernels": int(num_kernels),
        "mask_channels": bool(add_mask_channels),
        "target_row_count": {
            "support": _support(prior_samples["target_row_count"]),
            "distribution": _distribution(prior_samples["target_row_count"]),
            "stats": _safe_stats(prior_samples["target_row_count"].astype(np.float64)),
        },
        "train_target_std": _safe_stats(prior_samples["train_target_std"]),
        "test_target_std": _safe_stats(prior_samples["test_target_std"]),
        "low_variance_target_fraction": float(
            np.mean(prior_samples["test_target_std"] < 1e-3)
            if prior_samples["test_target_std"].size > 0
            else float("nan")
        ),
        "missing_mode_distribution": audit.get("family_distributions", {}).get(
            "missing_mode",
            {},
        ),
        "audit": audit,
        "feature_block_counts": _feature_block_counts(
            num_variables=prior_samples["num_variables"],
            explicit_lags=explicit_lags,
            num_kernels=num_kernels,
            add_mask_channels=add_mask_channels,
            deterministic_width=deterministic_width,
        ),
    }
    return summary, prior_samples


def summarize_benchmark(
    cfg: ForecastBenchmarkConfig,
    *,
    dataset_limit: int | None = None,
    series_limit: int | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    loaded_suite = load_suite(cfg)
    protocol = cfg.protocol
    benchmark_num_vars = 2
    active_feature_counts: list[int] = []
    feature_counts: list[int] = []
    context_rows: list[int] = []
    test_rows: list[int] = []
    target_rows: list[int] = []
    train_stds: list[float] = []
    test_stds: list[float] = []
    horizons: list[int] = []
    missing_fractions: list[float] = []
    processed_datasets = 0
    processed_series = 0
    skipped_datasets: list[dict[str, str]] = []

    for dataset_name in cfg.datasets.dataset_names:
        if dataset_limit is not None and processed_datasets >= dataset_limit:
            break
        bundle = loaded_suite[dataset_name]
        if bundle.skipped:
            skipped_datasets.append(
                {"dataset": dataset_name, "reason": bundle.skip_reason or "skipped"}
            )
            continue
        processed_datasets += 1
        for series_id, raw_series in enumerate(bundle.series):
            if series_limit is not None and series_id >= series_limit:
                break
            series = np.asarray(raw_series, dtype=np.float64)
            indices = generate_rolling_origin_indices(
                series_length=series.size,
                horizons=protocol.horizons,
                n_train=protocol.context_rows,
                n_test=protocol.test_rows,
                required_lag=protocol.required_lag,
                seed=cfg.seed + series_id,
            )
            table = build_forecast_table_from_series(
                cfg,
                series,
                t_idx=indices.t_idx,
                h_idx=indices.h_idx,
                split_index=indices.split_index,
                seed=cfg.seed + series_id,
            )
            split = int(table.split_index)
            x = np.asarray(table.x, dtype=np.float64)
            y = np.asarray(table.y, dtype=np.float64)
            feature_counts.append(int(x.shape[1]))
            active_feature_counts.append(int(np.any(np.abs(x) > 1e-12, axis=0).sum()))
            context_rows.append(split)
            test_rows.append(int(x.shape[0] - split))
            target_rows.append(int(y[split:].shape[0]))
            horizons.extend(int(v) for v in table.h_idx[split:].tolist())
            missing_fractions.append(float(np.mean(~np.isfinite(series))))
            train_target = y[:split]
            test_target = y[split:]
            train_stds.append(
                float(np.std(train_target, ddof=1 if train_target.size > 1 else 0))
            )
            test_stds.append(
                float(np.std(test_target, ddof=1 if test_target.size > 1 else 0))
            )
            processed_series += 1

    benchmark_samples = {
        "num_variables": np.full(
            (processed_series,), benchmark_num_vars, dtype=np.int64
        ),
        "context_rows": np.asarray(context_rows, dtype=np.int64),
        "test_rows": np.asarray(test_rows, dtype=np.int64),
        "horizons": np.asarray(horizons, dtype=np.int64),
        "feature_count_before_padding": np.asarray(feature_counts, dtype=np.int64),
        "active_feature_count": np.asarray(active_feature_counts, dtype=np.int64),
        "target_row_count": np.asarray(target_rows, dtype=np.int64),
        "train_target_std": np.asarray(train_stds, dtype=np.float64),
        "test_target_std": np.asarray(test_stds, dtype=np.float64),
        "missing_fraction": np.asarray(missing_fractions, dtype=np.float64),
    }
    deterministic_width = 1 + 1 + 1 + 2
    summary = {
        "source": "forecast_benchmark",
        "processed_datasets": int(processed_datasets),
        "processed_series": int(processed_series),
        "skipped_datasets": skipped_datasets,
        "num_variables": {
            "support": _support(benchmark_samples["num_variables"]),
            "distribution": _distribution(benchmark_samples["num_variables"]),
            "stats": _safe_stats(benchmark_samples["num_variables"].astype(np.float64)),
        },
        "context_rows": {
            "support": _support(benchmark_samples["context_rows"]),
            "distribution": _distribution(benchmark_samples["context_rows"]),
            "stats": _safe_stats(benchmark_samples["context_rows"].astype(np.float64)),
        },
        "test_rows": {
            "support": _support(benchmark_samples["test_rows"]),
            "distribution": _distribution(benchmark_samples["test_rows"]),
            "stats": _safe_stats(benchmark_samples["test_rows"].astype(np.float64)),
        },
        "horizons": {
            "support": _support(benchmark_samples["horizons"]),
            "distribution": _distribution(benchmark_samples["horizons"]),
        },
        "feature_count_before_padding": {
            "support": _support(benchmark_samples["feature_count_before_padding"]),
            "stats": _safe_stats(
                benchmark_samples["feature_count_before_padding"].astype(np.float64)
            ),
        },
        "active_feature_count": {
            "support": _support(benchmark_samples["active_feature_count"]),
            "stats": _safe_stats(
                benchmark_samples["active_feature_count"].astype(np.float64)
            ),
        },
        "lag_set": [int(v) for v in protocol.explicit_lags],
        "num_kernels": int(protocol.num_kernels),
        "mask_channels": bool(protocol.add_mask_channels),
        "target_row_count": {
            "support": _support(benchmark_samples["target_row_count"]),
            "distribution": _distribution(benchmark_samples["target_row_count"]),
            "stats": _safe_stats(
                benchmark_samples["target_row_count"].astype(np.float64)
            ),
        },
        "train_target_std": _safe_stats(benchmark_samples["train_target_std"]),
        "test_target_std": _safe_stats(benchmark_samples["test_target_std"]),
        "low_variance_target_fraction": float(
            np.mean(benchmark_samples["test_target_std"] < 1e-3)
            if benchmark_samples["test_target_std"].size > 0
            else float("nan")
        ),
        "missing_fraction": _safe_stats(benchmark_samples["missing_fraction"]),
        "feature_block_counts": _feature_block_counts(
            num_variables=benchmark_samples["num_variables"],
            explicit_lags=[int(v) for v in protocol.explicit_lags],
            num_kernels=int(protocol.num_kernels),
            add_mask_channels=bool(protocol.add_mask_channels),
            deterministic_width=deterministic_width,
        ),
    }
    return summary, benchmark_samples


def build_mismatch_report(
    prior_summary: dict[str, Any],
    prior_samples: dict[str, np.ndarray],
    benchmark_summary: dict[str, Any],
    benchmark_samples: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    prior_missing = cast(
        dict[str, float],
        prior_summary.get("missing_mode_distribution", {}),
    )
    benchmark_missing = {"observed_data": 1.0}
    mismatches = [
        {
            "dimension": "num_variables",
            "score": 1.0
            - _support_overlap(
                prior_summary["num_variables"]["support"],
                benchmark_summary["num_variables"]["support"],
            ),
            "prior": prior_summary["num_variables"]["support"],
            "benchmark": benchmark_summary["num_variables"]["support"],
        },
        {
            "dimension": "context_rows",
            "score": _empirical_wasserstein_1d(
                prior_samples["context_rows"],
                benchmark_samples["context_rows"],
            ),
            "prior": prior_summary["context_rows"]["stats"],
            "benchmark": benchmark_summary["context_rows"]["stats"],
        },
        {
            "dimension": "test_rows",
            "score": _empirical_wasserstein_1d(
                prior_samples["test_rows"],
                benchmark_samples["test_rows"],
            ),
            "prior": prior_summary["test_rows"]["stats"],
            "benchmark": benchmark_summary["test_rows"]["stats"],
        },
        {
            "dimension": "horizons",
            "score": 1.0
            - _support_overlap(
                prior_summary["horizons"]["support"],
                benchmark_summary["horizons"]["support"],
            ),
            "prior": prior_summary["horizons"]["support"],
            "benchmark": benchmark_summary["horizons"]["support"],
        },
        {
            "dimension": "feature_count_before_padding",
            "score": _empirical_wasserstein_1d(
                prior_samples["feature_count_before_padding"],
                benchmark_samples["feature_count_before_padding"],
            ),
            "prior": prior_summary["feature_count_before_padding"]["stats"],
            "benchmark": benchmark_summary["feature_count_before_padding"]["stats"],
        },
        {
            "dimension": "active_feature_count",
            "score": _empirical_wasserstein_1d(
                prior_samples["active_feature_count"],
                benchmark_samples["active_feature_count"],
            ),
            "prior": prior_summary["active_feature_count"]["stats"],
            "benchmark": benchmark_summary["active_feature_count"]["stats"],
        },
        {
            "dimension": "lag_set",
            "score": 1.0
            - _support_overlap(
                [int(v) for v in prior_summary["lag_set"]],
                [int(v) for v in benchmark_summary["lag_set"]],
            ),
            "prior": prior_summary["lag_set"],
            "benchmark": benchmark_summary["lag_set"],
        },
        {
            "dimension": "num_kernels",
            "score": float(
                abs(
                    int(prior_summary["num_kernels"])
                    - int(benchmark_summary["num_kernels"])
                )
            ),
            "prior": prior_summary["num_kernels"],
            "benchmark": benchmark_summary["num_kernels"],
        },
        {
            "dimension": "mask_channels",
            "score": float(
                bool(prior_summary["mask_channels"])
                != bool(benchmark_summary["mask_channels"])
            ),
            "prior": prior_summary["mask_channels"],
            "benchmark": benchmark_summary["mask_channels"],
        },
        {
            "dimension": "target_row_count",
            "score": _empirical_wasserstein_1d(
                prior_samples["target_row_count"],
                benchmark_samples["target_row_count"],
            ),
            "prior": prior_summary["target_row_count"]["stats"],
            "benchmark": benchmark_summary["target_row_count"]["stats"],
        },
        {
            "dimension": "train_target_std",
            "score": _empirical_wasserstein_1d(
                prior_samples["train_target_std"],
                benchmark_samples["train_target_std"],
            ),
            "prior": prior_summary["train_target_std"],
            "benchmark": benchmark_summary["train_target_std"],
        },
        {
            "dimension": "test_target_std",
            "score": _empirical_wasserstein_1d(
                prior_samples["test_target_std"],
                benchmark_samples["test_target_std"],
            ),
            "prior": prior_summary["test_target_std"],
            "benchmark": benchmark_summary["test_target_std"],
        },
        {
            "dimension": "low_variance_target_fraction",
            "score": abs(
                float(prior_summary["low_variance_target_fraction"])
                - float(benchmark_summary["low_variance_target_fraction"])
            ),
            "prior": prior_summary["low_variance_target_fraction"],
            "benchmark": benchmark_summary["low_variance_target_fraction"],
        },
        {
            "dimension": "missingness_pattern",
            "score": _total_variation_distance(prior_missing, benchmark_missing),
            "prior": prior_missing,
            "benchmark": benchmark_missing,
        },
    ]
    return sorted(
        mismatches,
        key=lambda item: (
            float(item["score"]) if np.isfinite(float(item["score"])) else -1.0
        ),
        reverse=True,
    )


def build_markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# DynSCM Prior vs Forecast Benchmark",
        "",
        "## Top Mismatches",
        "",
        "| Dimension | Score | Prior | Benchmark |",
        "| --- | ---: | --- | --- |",
    ]
    for mismatch in payload["mismatches"]:
        lines.append(
            "| {dimension} | {score:.4f} | `{prior}` | `{benchmark}` |".format(
                dimension=mismatch["dimension"],
                score=float(mismatch["score"]),
                prior=json.dumps(mismatch["prior"], sort_keys=True),
                benchmark=json.dumps(mismatch["benchmark"], sort_keys=True),
            )
        )
    return "\n".join(lines) + "\n"


def _load_benchmark_config(args: argparse.Namespace) -> ForecastBenchmarkConfig:
    cfg = (
        ForecastBenchmarkConfig.from_json(args.benchmark_config_json)
        if args.benchmark_config_json
        else ForecastBenchmarkConfig()
    )
    payload = cfg.to_dict()
    datasets_payload = dict(payload["datasets"])
    if args.cache_dir is not None:
        datasets_payload["cache_dir"] = Path(args.cache_dir)
    if args.allow_download is not None:
        datasets_payload["allow_download"] = bool(args.allow_download)
    payload["datasets"] = datasets_payload
    return ForecastBenchmarkConfig.from_dict(payload)


def main() -> None:
    args = _build_parser().parse_args()
    cfg = _load_benchmark_config(args)
    prior_summary, prior_samples = summarize_prior_dump(
        args.priordump,
        sample_limit=args.sample_limit,
    )
    benchmark_summary, benchmark_samples = summarize_benchmark(
        cfg,
        dataset_limit=args.dataset_limit,
        series_limit=args.series_limit,
    )
    mismatches = build_mismatch_report(
        prior_summary,
        prior_samples,
        benchmark_summary,
        benchmark_samples,
    )
    payload = {
        "prior_summary": prior_summary,
        "benchmark_summary": benchmark_summary,
        "mismatches": mismatches,
    }

    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_out is not None:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if args.markdown_out is not None:
        markdown_path = Path(args.markdown_out)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(build_markdown_report(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
