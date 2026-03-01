#!/usr/bin/env python3
"""Audit learnability and rejection diagnostics for a live DynSCM research profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import torch

import pretrain_regression_dynscm_live as live_train
from tfmplayground.priors.dynscm.research_profiles import (
    DynSCMLiveResearchProfile,
    LiveSourceSpec,
    get_research_profile,
)
from tfmplayground.utils import get_default_device


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research_profile", type=str, required=True)
    parser.add_argument("--source", type=str, default="train", choices=("train", "val"))
    parser.add_argument("--sample_steps", type=int, default=64)
    parser.add_argument("--batchsize", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dynscm_workers", type=int, default=4)
    parser.add_argument("--dynscm_worker_blas_threads", type=int, default=1)
    parser.add_argument("--json-out", type=str, default=None)
    return parser


def _source_for_name(
    profile: DynSCMLiveResearchProfile,
    source_name: str,
) -> LiveSourceSpec:
    if source_name == "train":
        return profile.train_source
    if source_name == "val":
        return profile.val_source
    raise ValueError(f"Unsupported source {source_name!r}.")


def _tensor(batch: dict[str, torch.Tensor | int], key: str) -> torch.Tensor:
    value = batch[key]
    if not torch.is_tensor(value):
        raise TypeError(f"Expected tensor batch field {key!r}.")
    return cast(torch.Tensor, value)


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "min": float("nan"),
            "median": float("nan"),
            "mean": float("nan"),
            "max": float("nan"),
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
    }


def _fraction_of(values: list[float], predicate) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(predicate(arr)))


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    profile = get_research_profile(args.research_profile)
    source = _source_for_name(profile, args.source)
    device = torch.device(args.device or get_default_device())
    batch_size = (
        int(args.batchsize)
        if args.batchsize is not None
        else int(profile.training_budget.batch_size)
    )
    loader = live_train._build_prior_loader(
        source=source,
        num_steps=int(args.sample_steps),
        batch_size=batch_size,
        num_datapoints_max=profile.max_seq_len,
        num_features=profile.max_features,
        device=device,
        seed=profile.train_seed if args.source == "train" else profile.val_seed,
        workers=int(args.dynscm_workers),
        worker_blas_threads=int(args.dynscm_worker_blas_threads),
        total_train_batches=max(1, int(args.sample_steps)),
    )

    train_target_std: list[float] = []
    test_target_std: list[float] = []
    max_abs_target_value: list[float] = []
    informative_feature_count: list[float] = []
    target_parent_count_native: list[float] = []
    target_parent_count_final: list[float] = []
    target_self_lag_weight_native: list[float] = []
    target_self_lag_weight_final: list[float] = []
    target_native_lag1_self_edge: list[float] = []
    probe_r2: list[float] = []
    missing_fraction: list[float] = []
    block_missing_fraction: list[float] = []
    forced_target_lag: list[float] = []
    forced_target_self_lag: list[float] = []
    mask_channels_enabled: list[float] = []
    clipped: list[float] = []
    low_std_reject: list[float] = []
    probe_r2_reject: list[float] = []
    clipped_reject: list[float] = []
    informative_feature_reject: list[float] = []
    missing_reject: list[float] = []
    attempts_used: list[float] = []

    try:
        for batch in loader:
            train_target_std.extend(
                _tensor(batch, "sampled_train_target_std")
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .tolist()
            )
            test_target_std.extend(
                _tensor(batch, "sampled_test_target_std")
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .tolist()
            )
            max_abs_target_value.extend(
                _tensor(batch, "sampled_max_abs_target_value")
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .tolist()
            )
            informative_feature_count.extend(
                _tensor(batch, "sampled_informative_feature_count")
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .tolist()
            )
            target_parent_count_native.extend(
                _tensor(batch, "sampled_target_parent_count_native")
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .tolist()
            )
            target_parent_count_final.extend(
                _tensor(batch, "sampled_target_parent_count_final")
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .tolist()
            )
            target_self_lag_weight_native.extend(
                _tensor(batch, "sampled_target_self_lag_weight_native")
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .tolist()
            )
            target_self_lag_weight_final.extend(
                _tensor(batch, "sampled_target_self_lag_weight_final")
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .tolist()
            )
            target_native_lag1_self_edge.extend(
                _tensor(batch, "sampled_target_native_lag1_self_edge")
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .tolist()
            )
            probe_r2.extend(
                _tensor(batch, "sampled_probe_r2")
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .tolist()
            )
            missing_fraction.extend(
                _tensor(batch, "sampled_missing_fraction")
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .tolist()
            )
            block_missing_fraction.extend(
                _tensor(batch, "sampled_block_missing_fraction")
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .tolist()
            )
            forced_target_lag.extend(
                _tensor(batch, "sampled_target_had_forced_lag_parent")
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .tolist()
            )
            forced_target_self_lag.extend(
                _tensor(batch, "sampled_target_had_forced_self_lag")
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .tolist()
            )
            mask_channels_enabled.extend(
                _tensor(batch, "sampled_mask_channels_enabled")
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .tolist()
            )
            clipped.extend(
                _tensor(batch, "sampled_simulation_clipped")
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .tolist()
            )
            low_std_reject.extend(
                _tensor(batch, "sampled_low_std_reject_count")
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .tolist()
            )
            probe_r2_reject.extend(
                _tensor(batch, "sampled_probe_r2_reject_count")
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .tolist()
            )
            clipped_reject.extend(
                _tensor(batch, "sampled_clipped_reject_count")
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .tolist()
            )
            informative_feature_reject.extend(
                _tensor(batch, "sampled_informative_feature_reject_count")
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .tolist()
            )
            missing_reject.extend(
                _tensor(batch, "sampled_missing_reject_count")
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .tolist()
            )
            attempts_used.extend(
                _tensor(batch, "sampled_generation_attempts_used")
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
                .tolist()
            )
    finally:
        close_fn = getattr(loader, "close", None)
        if callable(close_fn):
            close_fn()

    total_attempts = max(float(sum(attempts_used)), 1.0)
    finite_probe = [value for value in probe_r2 if np.isfinite(value)]
    payload = {
        "research_profile": profile.name,
        "source": args.source,
        "inspected_tables": len(train_target_std),
        "native_target_any_lag_parent_fraction": 1.0
        - _fraction_of(target_parent_count_native, lambda arr: arr <= 0.0),
        "native_target_lag1_self_edge_fraction": float(
            np.mean(target_native_lag1_self_edge)
        )
        if target_native_lag1_self_edge
        else 0.0,
        "target_parentless_fraction": _fraction_of(
            target_parent_count_final, lambda arr: arr <= 0.0
        ),
        "forced_target_lag_fraction": float(np.mean(forced_target_lag))
        if forced_target_lag
        else 0.0,
        "forced_target_self_lag_fraction": float(np.mean(forced_target_self_lag))
        if forced_target_self_lag
        else 0.0,
        "mask_channels_enabled_fraction": float(np.mean(mask_channels_enabled))
        if mask_channels_enabled
        else 0.0,
        "clipped_fraction": float(np.mean(clipped)) if clipped else 0.0,
        "train_target_std": _stats(train_target_std),
        "test_target_std": _stats(test_target_std),
        "max_abs_target_value": _stats(max_abs_target_value),
        "informative_feature_count": _stats(informative_feature_count),
        "native_target_parent_count_summary": _stats(target_parent_count_native),
        "final_target_parent_count_summary": _stats(target_parent_count_final),
        "native_target_self_lag_weight_summary": _stats(target_self_lag_weight_native),
        "final_target_self_lag_weight_summary": _stats(target_self_lag_weight_final),
        "target_self_lag_weight": _stats(target_self_lag_weight_final),
        "missing_fraction": _stats(missing_fraction),
        "block_missing_fraction": _stats(block_missing_fraction),
        "probe_r2": _stats(finite_probe),
        "rejections": {
            "low_std_fraction": float(sum(low_std_reject)) / total_attempts,
            "probe_r2_fraction": float(sum(probe_r2_reject)) / total_attempts,
            "clipped_fraction": float(sum(clipped_reject)) / total_attempts,
            "informative_feature_fraction": float(sum(informative_feature_reject))
            / total_attempts,
            "missing_fraction": float(sum(missing_reject)) / total_attempts,
        },
    }

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_out is not None:
        Path(args.json_out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
