"""Main module for the priors package."""

import argparse
import contextlib
import json
import random
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import torch

from tfmplayground.gcs_utils import is_gcs_uri, upload_local_file_to_gcs

from .dataloader import (
    DynSCMPriorDataLoader,
    TabICLPriorDataLoader,
    TICLPriorDataLoader,
)
from .dynscm import DynSCMConfig, dynscm_family_id_mappings
from .utils import build_tabpfn_prior, build_ticl_prior, dump_prior_to_h5

try:
    from .dataloader import TabPFNPriorDataLoader  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - legacy optional integration.
    TabPFNPriorDataLoader = None


_DYNSCM_PROFILES: dict[str, dict[str, object]] = {
    "default": {"cli_defaults": {}, "dynscm_overrides": []},
    "rich_t4_96x128": {
        "cli_defaults": {
            "num_batches": 4000,
            "batch_size": 8,
            "max_seq_len": 96,
            "max_features": 128,
            "max_classes": 0,
        },
        "dynscm_overrides": [
            "num_variables_min=4",
            "num_variables_max=10",
            "train_rows_min=24",
            "train_rows_max=72",
            "test_rows_min=8",
            "test_rows_max=24",
            "series_length_min=128",
            "series_length_max=512",
            "forecast_horizons=[1,2,3,5,8,14]",
            "num_regimes=5",
            "sticky_rho=0.90",
            "shared_order=false",
            "share_base_graph=false",
            "drift_std=0.02",
            "max_lag=20",
            "max_contemp_parents=4",
            "max_lagged_parents=4",
            "contemp_parent_rate=1.6",
            "lagged_parent_rate=2.0",
            "contemp_edge_add_prob=0.08",
            "contemp_edge_del_prob=0.08",
            "lagged_edge_add_prob=0.08",
            "lagged_edge_del_prob=0.08",
            'mechanism_type_choices=["linear_var","linear_plus_residual"]',
            "mechanism_type_probs=[0.35,0.65]",
            'noise_family_choices=["normal","student_t"]',
            "noise_family_probs=[0.40,0.60]",
            "student_df_min=3.5",
            "student_df_max=8.0",
            'missing_mode_choices=["off","mcar","mar","mnar_lite","mix"]',
            "missing_mode_probs=[0.05,0.20,0.20,0.20,0.35]",
            'kernel_family_choices=["exp_decay","power_law","mix"]',
            "kernel_family_probs=[0.25,0.25,0.50]",
            "explicit_lags=[0,1,2,5]",
            "num_kernels=2",
            "max_feature_lag=20",
            "add_mask_channels=true",
        ],
    },
    "benchmark_aligned_16k": {
        "cli_defaults": {
            "num_batches": 2000,
            "batch_size": 8,
            "max_seq_len": 48,
            "max_features": 64,
            "max_classes": 0,
        },
        "dynscm_overrides": [
            "num_variables_min=2",
            "num_variables_max=2",
            "train_rows_min=32",
            "train_rows_max=32",
            "test_rows_min=16",
            "test_rows_max=16",
            "series_length_min=96",
            "series_length_max=384",
            "forecast_horizons=[1,3,6,12]",
            "num_regimes=5",
            "sticky_rho=0.90",
            "shared_order=false",
            "share_base_graph=false",
            "drift_std=0.02",
            "max_lag=32",
            "max_contemp_parents=4",
            "max_lagged_parents=4",
            "contemp_parent_rate=1.6",
            "lagged_parent_rate=2.0",
            "contemp_edge_add_prob=0.08",
            "contemp_edge_del_prob=0.08",
            "lagged_edge_add_prob=0.08",
            "lagged_edge_del_prob=0.08",
            'mechanism_type_choices=["linear_var","linear_plus_residual"]',
            "mechanism_type_probs=[0.35,0.65]",
            'noise_family_choices=["normal","student_t"]',
            "noise_family_probs=[0.40,0.60]",
            "student_df_min=3.5",
            "student_df_max=8.0",
            'missing_mode_choices=["off","mcar","mar","mnar_lite","mix"]',
            "missing_mode_probs=[0.05,0.20,0.20,0.20,0.35]",
            'kernel_family_choices=["exp_decay","power_law","mix"]',
            "kernel_family_probs=[0.25,0.25,0.50]",
            "explicit_lags=[0,1,2,5,10]",
            "num_kernels=3",
            "max_feature_lag=32",
            "add_mask_channels=true",
        ],
    },
    "benchmark_aligned_easy_16k": {
        "cli_defaults": {
            "num_batches": 2000,
            "batch_size": 8,
            "max_seq_len": 48,
            "max_features": 64,
            "max_classes": 0,
        },
        "dynscm_overrides": [
            "num_variables_min=2",
            "num_variables_max=2",
            "train_rows_min=32",
            "train_rows_max=32",
            "test_rows_min=16",
            "test_rows_max=16",
            "series_length_min=128",
            "series_length_max=128",
            "forecast_horizons=[1,3,6,12]",
            "num_regimes=1",
            "sticky_rho=1.0",
            "shared_order=true",
            "share_base_graph=true",
            "drift_std=0.0",
            "max_lag=32",
            "use_contemp_edges=false",
            "max_contemp_parents=0",
            "max_lagged_parents=1",
            "contemp_parent_rate=0.0",
            "lagged_parent_rate=1.0",
            "contemp_edge_add_prob=0.0",
            "contemp_edge_del_prob=0.0",
            "lagged_edge_add_prob=0.0",
            "lagged_edge_del_prob=0.0",
            'mechanism_type="linear_var"',
            'noise_family="normal"',
            "noise_scale_min=0.02",
            "noise_scale_max=0.05",
            'missing_mode="off"',
            'kernel_family="exp_decay"',
            "explicit_lags=[0,1,2,5,10]",
            "num_kernels=3",
            "max_feature_lag=32",
            "add_mask_channels=true",
        ],
    },
    "benchmark_aligned_easy_plus_16k": {
        "cli_defaults": {
            "num_batches": 2000,
            "batch_size": 8,
            "max_seq_len": 48,
            "max_features": 64,
            "max_classes": 0,
        },
        "dynscm_overrides": [
            "num_variables_min=2",
            "num_variables_max=2",
            "train_rows_min=32",
            "train_rows_max=32",
            "test_rows_min=16",
            "test_rows_max=16",
            "series_length_min=128",
            "series_length_max=128",
            "forecast_horizons=[1,3,6,12]",
            "num_regimes=1",
            "sticky_rho=1.0",
            "shared_order=true",
            "share_base_graph=true",
            "drift_std=0.0",
            "max_lag=32",
            "use_contemp_edges=false",
            "max_contemp_parents=0",
            "max_lagged_parents=2",
            "contemp_parent_rate=0.0",
            "lagged_parent_rate=1.0",
            "contemp_edge_add_prob=0.0",
            "contemp_edge_del_prob=0.0",
            "lagged_edge_add_prob=0.0",
            "lagged_edge_del_prob=0.0",
            'mechanism_type_choices=["linear_var","linear_plus_residual"]',
            "mechanism_type_probs=[0.85,0.15]",
            'noise_family="normal"',
            "noise_scale_min=0.02",
            "noise_scale_max=0.06",
            'missing_mode="off"',
            'kernel_family_choices=["exp_decay","mix"]',
            "kernel_family_probs=[0.8,0.2]",
            "explicit_lags=[0,1,2,5,10]",
            "num_kernels=3",
            "max_feature_lag=32",
            "add_mask_channels=true",
        ],
    },
    # Retained as legacy exploratory profiles; the recommended automated
    # curriculum path now uses medium_graph/noise/missing instead.
    "benchmark_aligned_mechanism_16k": {
        "cli_defaults": {
            "num_batches": 2000,
            "batch_size": 8,
            "max_seq_len": 48,
            "max_features": 64,
            "max_classes": 0,
        },
        "dynscm_overrides": [
            "num_variables_min=2",
            "num_variables_max=2",
            "train_rows_min=32",
            "train_rows_max=32",
            "test_rows_min=16",
            "test_rows_max=16",
            "series_length_min=128",
            "series_length_max=128",
            "forecast_horizons=[1,3,6,12]",
            "num_regimes=1",
            "sticky_rho=1.0",
            "shared_order=true",
            "share_base_graph=true",
            "drift_std=0.0",
            "max_lag=32",
            "use_contemp_edges=false",
            "max_contemp_parents=0",
            "max_lagged_parents=2",
            "contemp_parent_rate=0.0",
            "lagged_parent_rate=1.0",
            "contemp_edge_add_prob=0.0",
            "contemp_edge_del_prob=0.0",
            "lagged_edge_add_prob=0.0",
            "lagged_edge_del_prob=0.0",
            'mechanism_type_choices=["linear_var","linear_plus_residual"]',
            "mechanism_type_probs=[0.7,0.3]",
            'noise_family="normal"',
            "noise_scale_min=0.02",
            "noise_scale_max=0.06",
            'missing_mode="off"',
            'kernel_family_choices=["exp_decay","mix"]',
            "kernel_family_probs=[0.8,0.2]",
            "explicit_lags=[0,1,2,5,10]",
            "num_kernels=3",
            "max_feature_lag=32",
            "add_mask_channels=true",
        ],
    },
    "benchmark_aligned_edges_soft_16k": {
        "cli_defaults": {
            "num_batches": 2000,
            "batch_size": 8,
            "max_seq_len": 48,
            "max_features": 64,
            "max_classes": 0,
        },
        # Adds only a small amount of structural graph drift (lagged edge
        # add/del) on top of the mechanism-mix stage.
        "dynscm_overrides": [
            "num_variables_min=2",
            "num_variables_max=2",
            "train_rows_min=32",
            "train_rows_max=32",
            "test_rows_min=16",
            "test_rows_max=16",
            "series_length_min=128",
            "series_length_max=128",
            "forecast_horizons=[1,3,6,12]",
            "num_regimes=1",
            "sticky_rho=1.0",
            "shared_order=true",
            "share_base_graph=true",
            "drift_std=0.0",
            "max_lag=32",
            "use_contemp_edges=false",
            "max_contemp_parents=0",
            "max_lagged_parents=2",
            "contemp_parent_rate=0.0",
            "lagged_parent_rate=1.0",
            "contemp_edge_add_prob=0.0",
            "contemp_edge_del_prob=0.0",
            "lagged_edge_add_prob=0.005",
            "lagged_edge_del_prob=0.005",
            'mechanism_type_choices=["linear_var","linear_plus_residual"]',
            "mechanism_type_probs=[0.7,0.3]",
            'noise_family="normal"',
            "noise_scale_min=0.02",
            "noise_scale_max=0.06",
            'missing_mode="off"',
            'kernel_family_choices=["exp_decay","mix"]',
            "kernel_family_probs=[0.8,0.2]",
            "explicit_lags=[0,1,2,5,10]",
            "num_kernels=3",
            "max_feature_lag=32",
            "add_mask_channels=true",
        ],
    },
    "benchmark_aligned_medium_graph_16k": {
        "cli_defaults": {
            "num_batches": 2000,
            "batch_size": 8,
            "max_seq_len": 48,
            "max_features": 64,
            "max_classes": 0,
        },
        "dynscm_overrides": [
            "num_variables_min=2",
            "num_variables_max=2",
            "train_rows_min=32",
            "train_rows_max=32",
            "test_rows_min=16",
            "test_rows_max=16",
            "series_length_min=128",
            "series_length_max=128",
            "forecast_horizons=[1,3,6,12]",
            "num_regimes=1",
            "sticky_rho=1.0",
            "shared_order=true",
            "share_base_graph=true",
            "drift_std=0.0",
            "max_lag=32",
            "use_contemp_edges=false",
            "max_contemp_parents=0",
            "max_lagged_parents=2",
            "contemp_parent_rate=0.0",
            "lagged_parent_rate=1.0",
            "contemp_edge_add_prob=0.0",
            "contemp_edge_del_prob=0.0",
            "lagged_edge_add_prob=0.02",
            "lagged_edge_del_prob=0.02",
            'mechanism_type_choices=["linear_var","linear_plus_residual"]',
            "mechanism_type_probs=[0.7,0.3]",
            'noise_family="normal"',
            "noise_scale_min=0.03",
            "noise_scale_max=0.08",
            'missing_mode="off"',
            'kernel_family_choices=["exp_decay","mix"]',
            "kernel_family_probs=[0.7,0.3]",
            "explicit_lags=[0,1,2,5,10]",
            "num_kernels=3",
            "max_feature_lag=32",
            "add_mask_channels=true",
        ],
    },
    "benchmark_aligned_medium_noise_16k": {
        "cli_defaults": {
            "num_batches": 2000,
            "batch_size": 8,
            "max_seq_len": 48,
            "max_features": 64,
            "max_classes": 0,
        },
        "dynscm_overrides": [
            "num_variables_min=2",
            "num_variables_max=2",
            "train_rows_min=32",
            "train_rows_max=32",
            "test_rows_min=16",
            "test_rows_max=16",
            "series_length_min=128",
            "series_length_max=128",
            "forecast_horizons=[1,3,6,12]",
            "num_regimes=1",
            "sticky_rho=1.0",
            "shared_order=true",
            "share_base_graph=true",
            "drift_std=0.0",
            "max_lag=32",
            "use_contemp_edges=false",
            "max_contemp_parents=0",
            "max_lagged_parents=2",
            "contemp_parent_rate=0.0",
            "lagged_parent_rate=1.0",
            "contemp_edge_add_prob=0.0",
            "contemp_edge_del_prob=0.0",
            "lagged_edge_add_prob=0.02",
            "lagged_edge_del_prob=0.02",
            'mechanism_type_choices=["linear_var","linear_plus_residual"]',
            "mechanism_type_probs=[0.7,0.3]",
            'noise_family_choices=["normal","student_t"]',
            "noise_family_probs=[0.8,0.2]",
            "student_df_min=4.0",
            "student_df_max=8.0",
            "noise_scale_min=0.03",
            "noise_scale_max=0.08",
            'missing_mode="off"',
            'kernel_family_choices=["exp_decay","mix"]',
            "kernel_family_probs=[0.7,0.3]",
            "explicit_lags=[0,1,2,5,10]",
            "num_kernels=3",
            "max_feature_lag=32",
            "add_mask_channels=true",
        ],
    },
    "benchmark_aligned_medium_missing_16k": {
        "cli_defaults": {
            "num_batches": 2000,
            "batch_size": 8,
            "max_seq_len": 48,
            "max_features": 64,
            "max_classes": 0,
        },
        "dynscm_overrides": [
            "num_variables_min=2",
            "num_variables_max=2",
            "train_rows_min=32",
            "train_rows_max=32",
            "test_rows_min=16",
            "test_rows_max=16",
            "series_length_min=128",
            "series_length_max=128",
            "forecast_horizons=[1,3,6,12]",
            "num_regimes=1",
            "sticky_rho=1.0",
            "shared_order=true",
            "share_base_graph=true",
            "drift_std=0.0",
            "max_lag=32",
            "use_contemp_edges=false",
            "max_contemp_parents=0",
            "max_lagged_parents=2",
            "contemp_parent_rate=0.0",
            "lagged_parent_rate=1.0",
            "contemp_edge_add_prob=0.0",
            "contemp_edge_del_prob=0.0",
            "lagged_edge_add_prob=0.02",
            "lagged_edge_del_prob=0.02",
            'mechanism_type_choices=["linear_var","linear_plus_residual"]',
            "mechanism_type_probs=[0.7,0.3]",
            'noise_family_choices=["normal","student_t"]',
            "noise_family_probs=[0.8,0.2]",
            "student_df_min=4.0",
            "student_df_max=8.0",
            "noise_scale_min=0.03",
            "noise_scale_max=0.08",
            'missing_mode_choices=["off","mcar"]',
            "missing_mode_probs=[0.8,0.2]",
            'kernel_family_choices=["exp_decay","mix"]',
            "kernel_family_probs=[0.7,0.3]",
            "explicit_lags=[0,1,2,5,10]",
            "num_kernels=3",
            "max_feature_lag=32",
            "add_mask_channels=true",
        ],
    },
    "benchmark_aligned_medium_32k": {
        "cli_defaults": {
            "num_batches": 4000,
            "batch_size": 8,
            "max_seq_len": 48,
            "max_features": 64,
            "max_classes": 0,
        },
        "dynscm_overrides": [
            "num_variables_min=2",
            "num_variables_max=2",
            "train_rows_min=32",
            "train_rows_max=32",
            "test_rows_min=16",
            "test_rows_max=16",
            "series_length_min=128",
            "series_length_max=256",
            "forecast_horizons=[1,3,6,12]",
            "num_regimes=2",
            "sticky_rho=0.95",
            "shared_order=true",
            "share_base_graph=true",
            "drift_std=0.01",
            "max_lag=32",
            "use_contemp_edges=false",
            "max_contemp_parents=0",
            "max_lagged_parents=2",
            "contemp_parent_rate=0.0",
            "lagged_parent_rate=1.0",
            "contemp_edge_add_prob=0.0",
            "contemp_edge_del_prob=0.0",
            "lagged_edge_add_prob=0.02",
            "lagged_edge_del_prob=0.02",
            'mechanism_type_choices=["linear_var","linear_plus_residual"]',
            "mechanism_type_probs=[0.7,0.3]",
            'noise_family_choices=["normal","student_t"]',
            "noise_family_probs=[0.8,0.2]",
            "student_df_min=4.0",
            "student_df_max=8.0",
            "noise_scale_min=0.03",
            "noise_scale_max=0.08",
            'missing_mode_choices=["off","mcar"]',
            "missing_mode_probs=[0.8,0.2]",
            'kernel_family_choices=["exp_decay","mix"]',
            "kernel_family_probs=[0.7,0.3]",
            "explicit_lags=[0,1,2,5,10]",
            "num_kernels=3",
            "max_feature_lag=32",
            "add_mask_channels=true",
        ],
    },
    "benchmark_aligned_32k": {
        "cli_defaults": {
            "num_batches": 4000,
            "batch_size": 8,
            "max_seq_len": 48,
            "max_features": 64,
            "max_classes": 0,
        },
        "dynscm_overrides": [
            "num_variables_min=2",
            "num_variables_max=2",
            "train_rows_min=32",
            "train_rows_max=32",
            "test_rows_min=16",
            "test_rows_max=16",
            "series_length_min=96",
            "series_length_max=384",
            "forecast_horizons=[1,3,6,12]",
            "num_regimes=5",
            "sticky_rho=0.90",
            "shared_order=false",
            "share_base_graph=false",
            "drift_std=0.02",
            "max_lag=32",
            "max_contemp_parents=4",
            "max_lagged_parents=4",
            "contemp_parent_rate=1.6",
            "lagged_parent_rate=2.0",
            "contemp_edge_add_prob=0.08",
            "contemp_edge_del_prob=0.08",
            "lagged_edge_add_prob=0.08",
            "lagged_edge_del_prob=0.08",
            'mechanism_type_choices=["linear_var","linear_plus_residual"]',
            "mechanism_type_probs=[0.35,0.65]",
            'noise_family_choices=["normal","student_t"]',
            "noise_family_probs=[0.40,0.60]",
            "student_df_min=3.5",
            "student_df_max=8.0",
            'missing_mode_choices=["off","mcar","mar","mnar_lite","mix"]',
            "missing_mode_probs=[0.05,0.20,0.20,0.20,0.35]",
            'kernel_family_choices=["exp_decay","power_law","mix"]',
            "kernel_family_probs=[0.25,0.25,0.50]",
            "explicit_lags=[0,1,2,5,10]",
            "num_kernels=3",
            "max_feature_lag=32",
            "add_mask_channels=true",
        ],
    },
    # Backward-compatible explicit name for the richest benchmark-aligned profile.
    "benchmark_aligned_full_32k": {
        "cli_defaults": {
            "num_batches": 4000,
            "batch_size": 8,
            "max_seq_len": 48,
            "max_features": 64,
            "max_classes": 0,
        },
        "dynscm_overrides": [
            "num_variables_min=2",
            "num_variables_max=2",
            "train_rows_min=32",
            "train_rows_max=32",
            "test_rows_min=16",
            "test_rows_max=16",
            "series_length_min=96",
            "series_length_max=384",
            "forecast_horizons=[1,3,6,12]",
            "num_regimes=5",
            "sticky_rho=0.90",
            "shared_order=false",
            "share_base_graph=false",
            "drift_std=0.02",
            "max_lag=32",
            "max_contemp_parents=4",
            "max_lagged_parents=4",
            "contemp_parent_rate=1.6",
            "lagged_parent_rate=2.0",
            "contemp_edge_add_prob=0.08",
            "contemp_edge_del_prob=0.08",
            "lagged_edge_add_prob=0.08",
            "lagged_edge_del_prob=0.08",
            'mechanism_type_choices=["linear_var","linear_plus_residual"]',
            "mechanism_type_probs=[0.35,0.65]",
            'noise_family_choices=["normal","student_t"]',
            "noise_family_probs=[0.40,0.60]",
            "student_df_min=3.5",
            "student_df_max=8.0",
            'missing_mode_choices=["off","mcar","mar","mnar_lite","mix"]',
            "missing_mode_probs=[0.05,0.20,0.20,0.20,0.35]",
            'kernel_family_choices=["exp_decay","power_law","mix"]',
            "kernel_family_probs=[0.25,0.25,0.50]",
            "explicit_lags=[0,1,2,5,10]",
            "num_kernels=3",
            "max_feature_lag=32",
            "add_mask_channels=true",
        ],
    },
}


def _cli_option_was_set(raw_argv: list[str], option_name: str) -> bool:
    flag = f"--{option_name}"
    return any(token == flag or token.startswith(f"{flag}=") for token in raw_argv)


def _parse_override_value(raw_value: str) -> object:
    lower = raw_value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower == "null":
        return None
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def _parse_dynscm_overrides(raw_overrides: list[str]) -> dict[str, object]:
    """Parse `key=value` CLI overrides into flat DynSCM config fields.

    Dotted keys such as `shape.num_variables_min=8` are accepted and reduced to
    their leaf field name (`num_variables_min`), matching DynSCM flat overrides.
    """
    overrides: dict[str, object] = {}
    for entry in raw_overrides:
        if "=" not in entry:
            raise ValueError(
                f"Each --dynscm_override must be in `key=value` format, got {entry!r}."
            )
        raw_key, raw_value = entry.split("=", maxsplit=1)
        key = raw_key.strip()
        if not key:
            raise ValueError(f"Invalid empty override key in {entry!r}.")
        leaf_key = key.rsplit(".", maxsplit=1)[-1]
        overrides[leaf_key] = _parse_override_value(raw_value.strip())
    return overrides


def _load_dynscm_config(
    config_json: str | None,
    raw_overrides: list[str],
) -> DynSCMConfig:
    cfg = DynSCMConfig.from_json(config_json) if config_json else DynSCMConfig()
    overrides = _parse_dynscm_overrides(raw_overrides)
    return cfg.with_overrides(**overrides) if overrides else cfg


def _write_dump_metadata(save_path: str, metadata: dict[str, object]) -> None:
    with h5py.File(save_path, "a") as f:
        if "dump_metadata_json" in f:
            del f["dump_metadata_json"]
        if "dump_schema_version" in f:
            del f["dump_schema_version"]
        f.create_dataset(
            "dump_schema_version",
            data="tfmplayground_prior_v2",
            dtype=h5py.string_dtype(),
        )
        f.create_dataset(
            "dump_metadata_json",
            data=json.dumps(metadata, sort_keys=True),
            dtype=h5py.string_dtype(),
        )


def main():
    parser = argparse.ArgumentParser(
        description="Dump prior data (TICL, TabICL, TabPFN, or DynSCM) into HDF5."
    )
    parser.add_argument(
        "--lib",
        type=str,
        required=True,
        choices=["ticl", "tabicl", "tabpfn", "dynscm"],
        help="Which library to use for the prior.",
    )
    parser.add_argument(
        "--save_path", type=str, required=False, help="Path to save the HDF5 file."
    )
    parser.add_argument(
        "--num_batches", type=int, default=100, help="Number of batches to dump."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for dumping."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run prior sampling on.",
    )
    parser.add_argument(
        "--prior_type",
        type=str,
        required=False,
        default=None,
        help=(
            "Type of prior to use. "
            "For TICL: mlp, gp, classification_adapter, "
            "boolean_conjunctions, step_function. "
            "For TabICL: mlp_scm, tree_scm, mix_scm, dummy. "
            "For TabPFN: mlp, gp, prior_bag."
        ),
    )
    parser.add_argument(
        "--base_prior",
        type=str,
        default="mlp",
        choices=["mlp", "gp"],
        help="Base regression prior for composite priors like classification_adapter.",
    )
    parser.add_argument(
        "--min_features", type=int, default=1, help="Minimum number of input features."
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=100,
        help="Maximum number of input features.",
    )
    parser.add_argument(
        "--min_seq_len",
        type=int,
        default=None,
        help="Minimum number of data points per function.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="Maximum number of data points per function.",
    )
    parser.add_argument(
        "--min_eval_pos",
        type=int,
        default=10,
        help="Minimum evaluation position in the sequence.",
    )
    parser.add_argument(
        "--max_classes",
        type=int,
        default=0,
        help=(
            "Maximum number of classes. Set to 0 for regression, >0 for classification."
        ),
    )
    parser.add_argument(
        "--np_seed", type=int, default=None, help="Random seed for NumPy."
    )
    parser.add_argument(
        "--torch_seed", type=int, default=None, help="Random seed for PyTorch."
    )
    parser.add_argument(
        "--dynscm_config_json",
        type=str,
        default=None,
        help=("Optional path to a DynSCM JSON config. Used only when --lib dynscm."),
    )
    parser.add_argument(
        "--dynscm_override",
        action="append",
        default=[],
        help=(
            "DynSCM override in key=value format. Can be repeated. "
            "Example: --dynscm_override num_variables_min=6 "
            "--dynscm_override features.num_kernels=2"
        ),
    )
    parser.add_argument(
        "--dynscm_profile",
        type=str,
        default="default",
        choices=sorted(_DYNSCM_PROFILES.keys()),
        help=(
            "Named DynSCM generation profile. Profile defaults are applied first "
            "and can be overridden by explicit CLI args / --dynscm_override."
        ),
    )
    parser.add_argument(
        "--dynscm_seed",
        type=int,
        default=None,
        help=(
            "Optional dedicated seed for DynSCM batch sampling. "
            "Defaults to --np_seed when omitted."
        ),
    )
    parser.add_argument(
        "--dynscm_workers",
        type=int,
        default=1,
        help="Number of process workers for DynSCM batch generation.",
    )
    parser.add_argument(
        "--dynscm_worker_blas_threads",
        type=int,
        default=1,
        help="BLAS thread cap applied inside each DynSCM worker process.",
    )
    parser.add_argument(
        "--dynscm_compute_spectral_diagnostics",
        dest="dynscm_compute_spectral_diagnostics",
        action="store_true",
        default=None,
        help=(
            "Compute spectral-radius diagnostics even when spectral rescaling "
            "is disabled."
        ),
    )
    parser.add_argument(
        "--no_dynscm_compute_spectral_diagnostics",
        dest="dynscm_compute_spectral_diagnostics",
        action="store_false",
        help="Disable optional spectral-radius diagnostics for DynSCM.",
    )

    raw_argv = sys.argv[1:]
    args = parser.parse_args(raw_argv)

    selected_dynscm_profile = _DYNSCM_PROFILES[args.dynscm_profile]
    profile_dynscm_overrides: list[str] = []
    if args.lib == "dynscm":
        profile_dynscm_overrides = list(
            selected_dynscm_profile.get("dynscm_overrides", [])
        )
        profile_cli_defaults = dict(selected_dynscm_profile.get("cli_defaults", {}))
        for option_name, option_value in profile_cli_defaults.items():
            if not _cli_option_was_set(raw_argv, option_name):
                setattr(args, option_name, option_value)

    if args.np_seed is not None:
        np.random.seed(args.np_seed)
    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)
        random.seed(args.torch_seed)

    if args.lib in {"ticl", "tabicl", "tabpfn"} and args.prior_type is None:
        parser.error("--prior_type is required for ticl/tabicl/tabpfn.")

    if args.lib == "dynscm" and args.max_classes != 0:
        parser.error("DynSCM currently supports regression only; use --max_classes 0.")
    if args.dynscm_workers < 1:
        parser.error("--dynscm_workers must be >= 1.")
    if args.dynscm_worker_blas_threads < 1:
        parser.error("--dynscm_worker_blas_threads must be >= 1.")

    device = torch.device(args.device)
    resolved_prior_type = args.prior_type if args.prior_type is not None else args.lib

    if args.save_path is None:
        args.save_path = (
            f"prior_{args.lib}_{resolved_prior_type}_{args.num_batches}"
            f"x{args.batch_size}_{args.max_seq_len}x{args.max_features}.h5"
        )

    # infer the problem_type from max_classes
    problem_type = "classification" if args.max_classes > 0 else "regression"
    dynscm_cfg: DynSCMConfig | None = None

    if args.lib == "ticl":
        prior = TICLPriorDataLoader(
            prior=build_ticl_prior(args.prior_type, args.base_prior, args.max_classes),
            num_steps=args.num_batches,
            batch_size=args.batch_size,
            num_datapoints_max=args.max_seq_len,
            num_features=args.max_features,
            device=device,
            min_eval_pos=args.min_eval_pos,
        )
    elif args.lib == "tabpfn":
        if TabPFNPriorDataLoader is None:
            raise RuntimeError(
                "TabPFNPriorDataLoader is unavailable in this installation."
            )
        tabpfn_config = build_tabpfn_prior(args.prior_type, args.max_classes)
        prior = TabPFNPriorDataLoader(
            prior_type=args.prior_type,
            num_steps=args.num_batches,
            batch_size=args.batch_size,
            num_datapoints_max=args.max_seq_len,
            num_features=args.max_features,
            device=device,
            **tabpfn_config,
        )
    elif args.lib == "dynscm":
        resolved_dynscm_overrides = profile_dynscm_overrides + list(
            args.dynscm_override
        )
        try:
            dynscm_cfg = _load_dynscm_config(
                args.dynscm_config_json,
                resolved_dynscm_overrides,
            )
        except ValueError as exc:
            parser.error(f"Invalid DynSCM configuration: {exc}")
        if args.dynscm_compute_spectral_diagnostics is not None:
            dynscm_cfg = dynscm_cfg.with_overrides(
                compute_spectral_diagnostics=args.dynscm_compute_spectral_diagnostics
            )

        dynscm_seed = args.dynscm_seed
        if dynscm_seed is None:
            dynscm_seed = args.np_seed

        prior = DynSCMPriorDataLoader(
            cfg=dynscm_cfg,
            num_steps=args.num_batches,
            batch_size=args.batch_size,
            num_datapoints_max=args.max_seq_len,
            num_features=args.max_features,
            device=device,
            seed=dynscm_seed,
            workers=args.dynscm_workers,
            worker_blas_threads=args.dynscm_worker_blas_threads,
        )
    else:  # tabicl
        prior = TabICLPriorDataLoader(
            num_steps=args.num_batches,
            batch_size=args.batch_size,
            num_datapoints_min=args.min_seq_len,
            num_datapoints_max=args.max_seq_len,
            min_features=args.min_features,
            max_features=args.max_features,
            max_num_classes=args.max_classes,
            prior_type=args.prior_type,
            device=device,
        )

    local_save_path = args.save_path
    temp_dir: Path | None = None
    if is_gcs_uri(args.save_path):
        temp_dir = Path(tempfile.mkdtemp(prefix="tfmplayground_prior_dump_"))
        local_save_path = str(temp_dir / Path(args.save_path).name)

    try:
        dump_prior_to_h5(
            prior,
            args.max_classes,
            args.batch_size,
            local_save_path,
            problem_type,
            args.max_seq_len,
            args.max_features,
        )
        metadata = {
            "created_utc": datetime.now(UTC).isoformat(),
            "schema_version": "tfmplayground_prior_v2",
            "lib": args.lib,
            "problem_type": problem_type,
            "prior_type": resolved_prior_type,
            "num_batches": int(args.num_batches),
            "batch_size": int(args.batch_size),
            "max_seq_len": int(args.max_seq_len),
            "max_features": int(args.max_features),
            "max_classes": int(args.max_classes),
            "np_seed": args.np_seed,
            "torch_seed": args.torch_seed,
            "dynscm_seed": args.dynscm_seed,
            "dynscm_workers": int(args.dynscm_workers),
            "dynscm_worker_blas_threads": int(args.dynscm_worker_blas_threads),
            "dynscm_profile": args.dynscm_profile,
            "dynscm_profile_overrides": profile_dynscm_overrides,
            "dynscm_overrides": (
                profile_dynscm_overrides + list(args.dynscm_override)
                if args.lib == "dynscm"
                else []
            ),
            "dynscm_user_overrides": list(args.dynscm_override),
            "dynscm_config_json": args.dynscm_config_json,
            "dynscm_config": dynscm_cfg.to_dict() if args.lib == "dynscm" else None,
            "dynscm_family_id_mappings": (
                dynscm_family_id_mappings() if args.lib == "dynscm" else None
            ),
        }
        # Test stubs may replace dump writing with non-HDF5 files.
        with contextlib.suppress(OSError):
            _write_dump_metadata(local_save_path, metadata)
        if is_gcs_uri(args.save_path):
            upload_local_file_to_gcs(local_save_path, args.save_path)
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
