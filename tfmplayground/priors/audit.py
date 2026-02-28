"""Audit helpers for prior dumps used during PFN pretraining."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import cast

import h5py
import numpy as np


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


def _shannon_entropy(probabilities: np.ndarray) -> float:
    if probabilities.size == 0:
        return float("nan")
    positive = probabilities[probabilities > 0.0]
    if positive.size == 0:
        return float("nan")
    return float(-np.sum(positive * np.log(positive)))


def _as_int(value: object) -> int:
    return int(cast(int | np.integer | str, value))


def _as_float(value: object) -> float:
    return float(cast(float | np.floating | int | np.integer | str, value))


def _as_mapping(value: object) -> Mapping[object, object] | None:
    if isinstance(value, Mapping):
        return value
    return None


def _load_dump_metadata_payload(h5_file: h5py.File) -> dict[str, object]:
    if "dump_metadata_json" not in h5_file:
        return {}
    raw_metadata = h5_file["dump_metadata_json"][()]
    if isinstance(raw_metadata, bytes):
        metadata_text = raw_metadata.decode("utf-8")
    else:
        metadata_text = str(raw_metadata)
    try:
        payload = json.loads(metadata_text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return cast(dict[str, object], payload)


def _load_family_name_mappings(h5_file: h5py.File) -> dict[str, dict[int, str]]:
    payload = _load_dump_metadata_payload(h5_file)
    raw_mappings = payload.get("dynscm_family_id_mappings")
    if not isinstance(raw_mappings, dict):
        return {}

    mappings: dict[str, dict[int, str]] = {}
    for family_name, mapping in raw_mappings.items():
        if not isinstance(family_name, str) or not isinstance(mapping, dict):
            continue
        parsed_mapping: dict[int, str] = {}
        for raw_key, raw_value in mapping.items():
            try:
                parsed_key = int(raw_key)
            except (TypeError, ValueError):
                continue
            parsed_mapping[parsed_key] = str(raw_value)
        if parsed_mapping:
            mappings[family_name] = parsed_mapping
    return mappings


def _support(values: np.ndarray) -> list[int]:
    if values.size == 0:
        return []
    return [int(v) for v in np.unique(values).tolist()]


def _feature_block_stats(values: np.ndarray) -> dict[str, float]:
    return _safe_stats(values.astype(np.float64))


def _deterministic_feature_count(dynscm_cfg: Mapping[object, object]) -> int:
    return (
        int(bool(dynscm_cfg.get("add_time_feature", True)))
        + int(bool(dynscm_cfg.get("add_horizon_feature", True)))
        + int(bool(dynscm_cfg.get("add_log_horizon", True)))
        + (2 if bool(dynscm_cfg.get("add_seasonality", True)) else 0)
    )


def _target_entropy(values: np.ndarray, *, bins: int = 64) -> float:
    if values.size == 0:
        return float("nan")
    clipped = np.clip(values.astype(np.float64), -6.0, 6.0)
    hist, _ = np.histogram(clipped, bins=bins, range=(-6.0, 6.0), density=False)
    total = int(np.sum(hist))
    if total == 0:
        return float("nan")
    probabilities = hist.astype(np.float64) / float(total)
    return _shannon_entropy(probabilities)


def audit_prior_dump(
    filename: str,
    *,
    sample_limit: int | None = None,
    nonzero_eps: float = 1e-12,
    chunk_size: int = 256,
    duplicate_round_decimals: int = 3,
) -> dict[str, object]:
    """Compute a structural audit for an HDF5 prior dump.

    The audit is intentionally conservative. It infers effective rows from
    non-zero feature content, which is robust for DynSCM dumps where padded rows
    are fully zeroed.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1.")

    with h5py.File(filename, "r") as f:
        num_functions, seq_len, feature_dim = f["X"].shape
        inspected = num_functions
        if sample_limit is not None:
            inspected = min(num_functions, max(1, int(sample_limit)))
        inspected_indices = np.arange(inspected, dtype=np.int64)
        metadata_payload = _load_dump_metadata_payload(f)
        family_name_mappings = _load_family_name_mappings(f)
        dynscm_cfg = _as_mapping(metadata_payload.get("dynscm_config"))

        has_num_datapoints = "num_datapoints" in f
        single_eval_pos = np.asarray(
            f["single_eval_pos"][inspected_indices],
            dtype=np.int64,
        )
        if has_num_datapoints:
            num_datapoints = np.asarray(
                f["num_datapoints"][inspected_indices],
                dtype=np.int64,
            )
        else:
            num_datapoints = np.full(
                (inspected,),
                seq_len,
                dtype=np.int64,
            )
        has_pre_budget_feature_count = "sampled_pre_budget_feature_count" in f
        if has_pre_budget_feature_count:
            pre_budget_feature_count = np.asarray(
                f["sampled_pre_budget_feature_count"][inspected_indices],
                dtype=np.int64,
            )
        else:
            pre_budget_feature_count = np.full((inspected,), -1, dtype=np.int64)
        sampled_num_vars = (
            np.asarray(f["sampled_num_vars"][inspected_indices], dtype=np.int64)
            if "sampled_num_vars" in f
            else np.full((inspected,), -1, dtype=np.int64)
        )
        sampled_n_train = (
            np.asarray(f["sampled_n_train"][inspected_indices], dtype=np.int64)
            if "sampled_n_train" in f
            else single_eval_pos.copy()
        )
        sampled_n_test = (
            np.asarray(f["sampled_n_test"][inspected_indices], dtype=np.int64)
            if "sampled_n_test" in f
            else np.clip(num_datapoints - single_eval_pos, 0, None)
        )

        target_rows = np.clip(num_datapoints - single_eval_pos, 0, None)
        num_datapoints_out_of_range = int(
            np.sum((num_datapoints < 1) | (num_datapoints > seq_len))
        )
        single_eval_pos_out_of_range = int(
            np.sum((single_eval_pos < 0) | (single_eval_pos >= num_datapoints))
        )
        nonpositive_target_rows = int(np.sum(target_rows <= 0))

        inferred_effective_rows = np.zeros((inspected,), dtype=np.int64)
        active_feature_counts = np.zeros((inspected,), dtype=np.int64)
        train_y_std: list[float] = []
        target_y_std: list[float] = []
        normalized_target_values: list[np.ndarray] = []
        exact_hashes: set[bytes] = set()
        near_hashes: set[bytes] = set()
        family_id_datasets = {
            "mechanism_type": "sampled_mechanism_type_id",
            "noise_family": "sampled_noise_family_id",
            "missing_mode": "sampled_missing_mode_id",
            "kernel_family": "sampled_kernel_family_id",
        }
        family_ids: dict[str, np.ndarray] = {}
        for family_name, dataset_name in family_id_datasets.items():
            if dataset_name in f:
                family_ids[family_name] = np.asarray(
                    f[dataset_name][inspected_indices],
                    dtype=np.int64,
                )

        y_ds = f["y"]
        x_ds = f["X"]
        for start in range(0, inspected, chunk_size):
            end = min(start + chunk_size, inspected)
            row_ids = inspected_indices[start:end]
            x_chunk = np.asarray(x_ds[row_ids, :, :], dtype=np.float32)
            y_chunk = np.asarray(y_ds[row_ids, :], dtype=np.float32)
            sep_chunk = single_eval_pos[start:end]
            nd_chunk = num_datapoints[start:end]

            nonzero_rows = np.any(np.abs(x_chunk) > nonzero_eps, axis=2)
            inferred_chunk = nonzero_rows.sum(axis=1).astype(np.int64)
            inferred_effective_rows[start:end] = inferred_chunk

            for local_idx in range(end - start):
                inferred_rows = int(max(0, min(inferred_chunk[local_idx], seq_len)))
                if inferred_rows > 0:
                    active_feature_counts[start + local_idx] = int(
                        np.any(
                            np.abs(x_chunk[local_idx, :inferred_rows, :]) > nonzero_eps,
                            axis=0,
                        ).sum()
                    )
                nd_i = int(max(0, min(nd_chunk[local_idx], seq_len)))
                sep_i = int(max(0, min(sep_chunk[local_idx], nd_i)))
                train_target = y_chunk[local_idx, :sep_i]
                test_target = y_chunk[local_idx, sep_i:nd_i]
                x_row = x_chunk[local_idx, :nd_i, :]
                y_row = y_chunk[local_idx, :nd_i]
                exact_hashes.add(
                    hashlib.blake2b(
                        x_row.tobytes()
                        + y_row.tobytes()
                        + np.asarray([sep_i, nd_i], dtype=np.int32).tobytes(),
                        digest_size=16,
                    ).digest()
                )
                near_hashes.add(
                    hashlib.blake2b(
                        np.round(x_row, duplicate_round_decimals)
                        .astype(np.float32, copy=False)
                        .tobytes()
                        + np.round(y_row, duplicate_round_decimals)
                        .astype(np.float32, copy=False)
                        .tobytes()
                        + np.asarray([sep_i, nd_i], dtype=np.int32).tobytes(),
                        digest_size=16,
                    ).digest()
                )
                if train_target.size > 0:
                    ddof = 1 if train_target.size > 1 else 0
                    train_std = float(np.std(train_target, ddof=ddof))
                    train_y_std.append(train_std)
                if test_target.size > 0:
                    ddof = 1 if test_target.size > 1 else 0
                    target_y_std.append(float(np.std(test_target, ddof=ddof)))
                if train_target.size > 0 and test_target.size > 0:
                    train_mean = float(np.mean(train_target))
                    train_scale = train_std if train_std > 1e-8 else 1.0
                    normalized_target_values.append(
                        (test_target.astype(np.float64) - train_mean) / train_scale
                    )

    inferred_padding_rows = np.clip(num_datapoints - inferred_effective_rows, 0, None)
    inferred_padded_target_rows = np.minimum(inferred_padding_rows, target_rows)
    total_target_rows = int(target_rows.sum())
    padded_target_fraction = float(inferred_padded_target_rows.sum()) / max(
        1, total_target_rows
    )

    num_datapoints_stats = _safe_stats(num_datapoints.astype(np.float64))
    single_eval_pos_stats = _safe_stats(single_eval_pos.astype(np.float64))
    target_rows_stats = _safe_stats(target_rows.astype(np.float64))
    inferred_rows_stats = _safe_stats(inferred_effective_rows.astype(np.float64))
    active_features_stats = _safe_stats(active_feature_counts.astype(np.float64))
    train_std_stats = _safe_stats(np.asarray(train_y_std, dtype=np.float64))
    target_std_stats = _safe_stats(np.asarray(target_y_std, dtype=np.float64))
    pre_budget_stats = _safe_stats(pre_budget_feature_count.astype(np.float64))
    feature_truncation_fraction = float("nan")
    if has_pre_budget_feature_count:
        feature_truncation_fraction = float(
            np.mean(pre_budget_feature_count > feature_dim)
        )
    duplicate_fraction = 1.0 - (len(exact_hashes) / max(1, inspected))
    near_duplicate_fraction = 1.0 - (len(near_hashes) / max(1, inspected))
    effective_target_entropy = _target_entropy(
        np.concatenate(normalized_target_values, axis=0)
        if normalized_target_values
        else np.asarray([], dtype=np.float64)
    )

    configured_horizon_support: list[int] = []
    configured_explicit_lags: list[int] = []
    configured_num_kernels: int | None = None
    configured_max_feature_lag: int | None = None
    configured_add_mask_channels: bool | None = None
    feature_block_counts: dict[str, dict[str, float]] = {}
    if dynscm_cfg is not None:
        configured_horizon_support = [
            int(v) for v in cast(list[object], dynscm_cfg.get("forecast_horizons", []))
        ]
        configured_explicit_lags = [
            int(v) for v in cast(list[object], dynscm_cfg.get("explicit_lags", []))
        ]
        configured_num_kernels = int(cast(int, dynscm_cfg.get("num_kernels", 0)))
        configured_max_feature_lag = int(
            cast(int, dynscm_cfg.get("max_feature_lag", 0))
        )
        configured_add_mask_channels = bool(dynscm_cfg.get("add_mask_channels", False))
        valid_num_vars = sampled_num_vars[sampled_num_vars >= 0]
        if valid_num_vars.size > 0:
            lag_width = valid_num_vars * len(configured_explicit_lags)
            kernel_width = valid_num_vars * int(configured_num_kernels)
            data_width = lag_width + kernel_width
            mask_width = (
                data_width
                if configured_add_mask_channels
                else np.zeros_like(data_width)
            )
            deterministic_width = np.full(
                valid_num_vars.shape,
                _deterministic_feature_count(dynscm_cfg),
                dtype=np.int64,
            )
            total_width = data_width + mask_width + deterministic_width
            feature_block_counts = {
                "deterministic": _feature_block_stats(deterministic_width),
                "lag_values": _feature_block_stats(lag_width),
                "kernel_values": _feature_block_stats(kernel_width),
                "data_mask": _feature_block_stats(mask_width),
                "total_pre_budget_theoretical": _feature_block_stats(total_width),
            }

    family_distributions: dict[str, dict[str, float]] = {}
    family_entropies: dict[str, float] = {}
    for family_name, ids in family_ids.items():
        if ids.size == 0:
            continue
        unique_ids, counts = np.unique(ids, return_counts=True)
        fractions = counts.astype(np.float64) / float(np.sum(counts))
        labels = family_name_mappings.get(family_name, {})
        distribution: dict[str, float] = {}
        for family_id, fraction in zip(
            unique_ids.tolist(), fractions.tolist(), strict=True
        ):
            label = labels.get(int(family_id), str(int(family_id)))
            distribution[label] = float(fraction)
        family_distributions[family_name] = distribution
        family_entropies[family_name] = _shannon_entropy(fractions)

    return {
        "num_functions": int(num_functions),
        "inspected_functions": int(inspected),
        "seq_len": int(seq_len),
        "feature_dim": int(feature_dim),
        "has_num_datapoints_dataset": bool(has_num_datapoints),
        "num_datapoints_out_of_range_count": int(num_datapoints_out_of_range),
        "single_eval_pos_out_of_range_count": int(single_eval_pos_out_of_range),
        "nonpositive_target_rows_count": int(nonpositive_target_rows),
        "num_datapoints_min": num_datapoints_stats["min"],
        "num_datapoints_mean": num_datapoints_stats["mean"],
        "num_datapoints_median": num_datapoints_stats["median"],
        "num_datapoints_max": num_datapoints_stats["max"],
        "single_eval_pos_min": single_eval_pos_stats["min"],
        "single_eval_pos_mean": single_eval_pos_stats["mean"],
        "single_eval_pos_median": single_eval_pos_stats["median"],
        "single_eval_pos_max": single_eval_pos_stats["max"],
        "target_rows_min": target_rows_stats["min"],
        "target_rows_mean": target_rows_stats["mean"],
        "target_rows_median": target_rows_stats["median"],
        "target_rows_max": target_rows_stats["max"],
        "inferred_effective_rows_min": inferred_rows_stats["min"],
        "inferred_effective_rows_mean": inferred_rows_stats["mean"],
        "inferred_effective_rows_median": inferred_rows_stats["median"],
        "inferred_effective_rows_max": inferred_rows_stats["max"],
        "inferred_num_datapoints_mismatch_fraction": float(
            np.mean(inferred_effective_rows < num_datapoints)
        ),
        "inferred_padded_target_fraction": float(padded_target_fraction),
        "active_features_min": active_features_stats["min"],
        "active_features_mean": active_features_stats["mean"],
        "active_features_median": active_features_stats["median"],
        "active_features_max": active_features_stats["max"],
        "feature_budget_saturation_fraction": float(
            np.mean(active_feature_counts >= feature_dim)
        ),
        "has_pre_budget_feature_count_dataset": bool(has_pre_budget_feature_count),
        "pre_budget_feature_count_min": pre_budget_stats["min"],
        "pre_budget_feature_count_mean": pre_budget_stats["mean"],
        "pre_budget_feature_count_median": pre_budget_stats["median"],
        "pre_budget_feature_count_max": pre_budget_stats["max"],
        "feature_truncation_fraction": feature_truncation_fraction,
        "duplicate_fraction": float(duplicate_fraction),
        "near_duplicate_fraction": float(near_duplicate_fraction),
        "effective_target_entropy": float(effective_target_entropy),
        "num_vars_support": _support(sampled_num_vars[sampled_num_vars >= 0]),
        "n_train_support": _support(sampled_n_train),
        "n_test_support": _support(sampled_n_test),
        "horizon_support": configured_horizon_support,
        "configured_explicit_lags": configured_explicit_lags,
        "configured_num_kernels": configured_num_kernels,
        "configured_max_feature_lag": configured_max_feature_lag,
        "configured_add_mask_channels": configured_add_mask_channels,
        "feature_block_counts": feature_block_counts,
        "has_variant_family_metadata": bool(len(family_distributions) > 0),
        "family_distributions": family_distributions,
        "family_entropies": family_entropies,
        "train_y_std_min": train_std_stats["min"],
        "train_y_std_mean": train_std_stats["mean"],
        "train_y_std_median": train_std_stats["median"],
        "train_y_std_max": train_std_stats["max"],
        "target_y_std_min": target_std_stats["min"],
        "target_y_std_mean": target_std_stats["mean"],
        "target_y_std_median": target_std_stats["median"],
        "target_y_std_max": target_std_stats["max"],
    }


def integrity_errors(
    audit: dict[str, object],
    *,
    max_padded_target_fraction: float = 0.05,
    max_num_datapoints_mismatch_fraction: float = 0.01,
    min_family_fraction: float = 0.05,
    max_feature_truncation_fraction: float = 0.40,
    min_diversity_sample_size: int = 1000,
) -> list[str]:
    """Return hard-fail integrity issues for training."""
    issues: list[str] = []
    if not bool(audit["has_num_datapoints_dataset"]):
        issues.append("missing required num_datapoints dataset")
    if _as_int(audit["num_datapoints_out_of_range_count"]) > 0:
        issues.append("num_datapoints contains values outside [1, seq_len]")
    if _as_int(audit["single_eval_pos_out_of_range_count"]) > 0:
        issues.append("single_eval_pos contains values outside [0, num_datapoints)")
    if _as_int(audit["nonpositive_target_rows_count"]) > 0:
        issues.append("one or more rows have zero/negative target rows")
    if _as_float(audit["inferred_padded_target_fraction"]) > max_padded_target_fraction:
        issues.append(
            "inferred padded target fraction exceeds threshold "
            f"{max_padded_target_fraction:.3f}"
        )
    inspected = _as_int(audit["inspected_functions"])
    if (
        inspected >= 100
        and _as_float(audit["inferred_num_datapoints_mismatch_fraction"])
        > max_num_datapoints_mismatch_fraction
    ):
        issues.append(
            "inferred num_datapoints mismatch fraction exceeds threshold "
            f"{max_num_datapoints_mismatch_fraction:.3f}"
        )
    if bool(audit.get("has_variant_family_metadata")) and inspected >= int(
        min_diversity_sample_size
    ):
        family_distributions = _as_mapping(audit.get("family_distributions"))
        if family_distributions is not None:
            for family_name, distribution in family_distributions.items():
                typed_distribution = _as_mapping(distribution)
                if typed_distribution is None:
                    continue
                for label, fraction in typed_distribution.items():
                    value = _as_float(fraction)
                    if value < min_family_fraction:
                        issues.append(
                            "family coverage below threshold "
                            f"{min_family_fraction:.3f} for {family_name}:{label} "
                            f"(observed {value:.3f})"
                        )
        truncation_fraction = audit.get("feature_truncation_fraction")
        if truncation_fraction is not None:
            truncation_fraction_value = _as_float(truncation_fraction)
            if (
                np.isfinite(truncation_fraction_value)
                and truncation_fraction_value > max_feature_truncation_fraction
            ):
                issues.append(
                    "feature truncation fraction exceeds threshold "
                    f"{max_feature_truncation_fraction:.3f}"
                )
    return issues
