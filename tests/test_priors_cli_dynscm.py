"""CLI roundtrip test for DynSCM prior dump integration."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

from tfmplayground.priors.dataloader import PriorDumpDataLoader


def test_dynscm_cli_roundtrip_to_h5_and_loader_iteration(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "dynscm_roundtrip.h5"

    cmd = [
        sys.executable,
        "-m",
        "tfmplayground.priors",
        "--lib",
        "dynscm",
        "--num_batches",
        "1",
        "--batch_size",
        "2",
        "--max_seq_len",
        "12",
        "--max_features",
        "24",
        "--max_classes",
        "0",
        "--np_seed",
        "17",
        "--dynscm_seed",
        "17",
        "--dynscm_workers",
        "1",
        "--dynscm_worker_blas_threads",
        "1",
        "--no_dynscm_compute_spectral_diagnostics",
        "--dynscm_override",
        "num_variables_min=3",
        "--dynscm_override",
        "num_variables_max=3",
        "--dynscm_override",
        "series_length_min=64",
        "--dynscm_override",
        "series_length_max=64",
        "--dynscm_override",
        "max_lag=4",
        "--dynscm_override",
        "max_feature_lag=8",
        "--dynscm_override",
        "explicit_lags=[0,1,2]",
        "--dynscm_override",
        "forecast_horizons=[1,2,3]",
        "--dynscm_override",
        "train_rows_min=6",
        "--dynscm_override",
        "train_rows_max=6",
        "--dynscm_override",
        "test_rows_min=3",
        "--dynscm_override",
        "test_rows_max=3",
        "--dynscm_override",
        'mechanism_type="linear_var"',
        "--dynscm_override",
        "num_kernels=0",
        "--save_path",
        str(output_path),
    ]
    completed = subprocess.run(
        cmd,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert output_path.exists()
    assert output_path.stat().st_size > 0
    with h5py.File(output_path, "r") as f:
        assert "dump_schema_version" in f
        assert "dump_metadata_json" in f
        raw_metadata = f["dump_metadata_json"][()]
        metadata_text = (
            raw_metadata.decode("utf-8")
            if isinstance(raw_metadata, bytes)
            else str(raw_metadata)
        )
        metadata = json.loads(metadata_text)
        assert "dynscm_family_id_mappings" in metadata
        for dataset_name in (
            "sampled_mechanism_type_id",
            "sampled_noise_family_id",
            "sampled_missing_mode_id",
            "sampled_kernel_family_id",
            "sampled_student_df",
            "sampled_num_vars",
            "sampled_num_steps",
            "sampled_n_train",
            "sampled_n_test",
            "sampled_pre_budget_feature_count",
        ):
            assert dataset_name in f
            assert f[dataset_name].shape[0] == 2

    prior = PriorDumpDataLoader(
        filename=str(output_path),
        num_steps=1,
        batch_size=2,
        device=torch.device("cpu"),
    )
    batch = next(iter(prior))

    assert set(batch) == {
        "x",
        "y",
        "target_y",
        "single_eval_pos",
        "num_datapoints",
        "target_mask",
    }
    assert batch["x"].shape[0] == 2
    assert batch["y"].shape[0] == 2
    assert batch["target_y"].shape[0] == 2
    assert batch["target_mask"].shape[0] == 2

    split = int(batch["single_eval_pos"])
    num_datapoints = int(batch["num_datapoints"])
    assert 0 < split < batch["x"].shape[1]
    assert split < num_datapoints <= batch["x"].shape[1]
    assert torch.all(batch["target_mask"][:, :split] == 0)
    assert torch.all(batch["target_mask"][:, split:num_datapoints] == 1)
    assert torch.all(batch["target_mask"][:, num_datapoints:] == 0)
    assert torch.isfinite(batch["x"]).all()
    assert torch.isfinite(batch["y"]).all()
    assert not np.isnan(batch["y"][:, :split].numpy()).any()


def test_dynscm_cli_rich_profile_applies_profile_overrides(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "dynscm_rich_profile.h5"

    cmd = [
        sys.executable,
        "-m",
        "tfmplayground.priors",
        "--lib",
        "dynscm",
        "--dynscm_profile",
        "rich_t4_96x128",
        "--num_batches",
        "1",
        "--batch_size",
        "1",
        "--max_classes",
        "0",
        "--np_seed",
        "23",
        "--dynscm_seed",
        "23",
        "--dynscm_workers",
        "1",
        "--dynscm_worker_blas_threads",
        "1",
        "--no_dynscm_compute_spectral_diagnostics",
        "--save_path",
        str(output_path),
    ]
    subprocess.run(
        cmd,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    with h5py.File(output_path, "r") as f:
        raw_metadata = f["dump_metadata_json"][()]
        metadata_text = (
            raw_metadata.decode("utf-8")
            if isinstance(raw_metadata, bytes)
            else str(raw_metadata)
        )
        metadata = json.loads(metadata_text)
    assert metadata["dynscm_profile"] == "rich_t4_96x128"
    assert metadata["dynscm_config"]["num_variables_max"] == 10
    assert metadata["dynscm_config"]["max_lag"] == 20
    assert metadata["dynscm_config"]["mechanism_type_choices"] == [
        "linear_var",
        "linear_plus_residual",
    ]


def test_dynscm_cli_benchmark_aligned_profile_matches_benchmark_contract(
    tmp_path: Path,
):
    repo_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "dynscm_benchmark_aligned_profile.h5"

    cmd = [
        sys.executable,
        "-m",
        "tfmplayground.priors",
        "--lib",
        "dynscm",
        "--dynscm_profile",
        "benchmark_aligned_16k",
        "--num_batches",
        "1",
        "--batch_size",
        "1",
        "--max_classes",
        "0",
        "--np_seed",
        "29",
        "--dynscm_seed",
        "29",
        "--dynscm_workers",
        "1",
        "--dynscm_worker_blas_threads",
        "1",
        "--no_dynscm_compute_spectral_diagnostics",
        "--save_path",
        str(output_path),
    ]
    subprocess.run(
        cmd,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    with h5py.File(output_path, "r") as f:
        raw_metadata = f["dump_metadata_json"][()]
        metadata_text = (
            raw_metadata.decode("utf-8")
            if isinstance(raw_metadata, bytes)
            else str(raw_metadata)
        )
        metadata = json.loads(metadata_text)

    assert metadata["dynscm_profile"] == "benchmark_aligned_16k"
    assert metadata["max_seq_len"] == 48
    assert metadata["max_features"] == 64
    assert metadata["dynscm_config"]["num_variables_min"] == 2
    assert metadata["dynscm_config"]["num_variables_max"] == 2
    assert metadata["dynscm_config"]["train_rows_min"] == 32
    assert metadata["dynscm_config"]["train_rows_max"] == 32
    assert metadata["dynscm_config"]["test_rows_min"] == 16
    assert metadata["dynscm_config"]["test_rows_max"] == 16
    assert metadata["dynscm_config"]["forecast_horizons"] == [1, 3, 6, 12]
    assert metadata["dynscm_config"]["explicit_lags"] == [0, 1, 2, 5, 10]
    assert metadata["dynscm_config"]["num_kernels"] == 3
    assert metadata["dynscm_config"]["max_feature_lag"] == 32
    assert metadata["dynscm_config"]["add_mask_channels"] is True


def test_dynscm_cli_benchmark_aligned_easy_profile_is_deliberately_low_diversity(
    tmp_path: Path,
):
    repo_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "dynscm_benchmark_aligned_easy_profile.h5"

    cmd = [
        sys.executable,
        "-m",
        "tfmplayground.priors",
        "--lib",
        "dynscm",
        "--dynscm_profile",
        "benchmark_aligned_easy_16k",
        "--num_batches",
        "1",
        "--batch_size",
        "1",
        "--max_classes",
        "0",
        "--np_seed",
        "31",
        "--dynscm_seed",
        "31",
        "--dynscm_workers",
        "1",
        "--dynscm_worker_blas_threads",
        "1",
        "--no_dynscm_compute_spectral_diagnostics",
        "--save_path",
        str(output_path),
    ]
    subprocess.run(
        cmd,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    with h5py.File(output_path, "r") as f:
        raw_metadata = f["dump_metadata_json"][()]
        metadata_text = (
            raw_metadata.decode("utf-8")
            if isinstance(raw_metadata, bytes)
            else str(raw_metadata)
        )
        metadata = json.loads(metadata_text)

    assert metadata["dynscm_profile"] == "benchmark_aligned_easy_16k"
    assert metadata["max_seq_len"] == 48
    assert metadata["max_features"] == 64
    assert metadata["dynscm_config"]["num_variables_min"] == 2
    assert metadata["dynscm_config"]["num_variables_max"] == 2
    assert metadata["dynscm_config"]["train_rows_min"] == 32
    assert metadata["dynscm_config"]["train_rows_max"] == 32
    assert metadata["dynscm_config"]["test_rows_min"] == 16
    assert metadata["dynscm_config"]["test_rows_max"] == 16
    assert metadata["dynscm_config"]["forecast_horizons"] == [1, 3, 6, 12]
    assert metadata["dynscm_config"]["explicit_lags"] == [0, 1, 2, 5, 10]
    assert metadata["dynscm_config"]["num_kernels"] == 3
    assert metadata["dynscm_config"]["num_regimes"] == 1
    assert metadata["dynscm_config"]["use_contemp_edges"] is False
    assert metadata["dynscm_config"]["mechanism_type"] == "linear_var"
    assert metadata["dynscm_config"]["noise_family"] == "normal"
    assert metadata["dynscm_config"]["missing_mode"] == "off"
    assert metadata["dynscm_config"]["kernel_family"] == "exp_decay"


def test_dynscm_cli_benchmark_aligned_easy_plus_profile_adds_only_small_diversity(
    tmp_path: Path,
):
    repo_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "dynscm_benchmark_aligned_easy_plus_profile.h5"

    cmd = [
        sys.executable,
        "-m",
        "tfmplayground.priors",
        "--lib",
        "dynscm",
        "--dynscm_profile",
        "benchmark_aligned_easy_plus_16k",
        "--num_batches",
        "1",
        "--batch_size",
        "1",
        "--max_classes",
        "0",
        "--np_seed",
        "33",
        "--dynscm_seed",
        "33",
        "--dynscm_workers",
        "1",
        "--dynscm_worker_blas_threads",
        "1",
        "--no_dynscm_compute_spectral_diagnostics",
        "--save_path",
        str(output_path),
    ]
    subprocess.run(
        cmd,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    with h5py.File(output_path, "r") as f:
        raw_metadata = f["dump_metadata_json"][()]
        metadata_text = (
            raw_metadata.decode("utf-8")
            if isinstance(raw_metadata, bytes)
            else str(raw_metadata)
        )
        metadata = json.loads(metadata_text)

    assert metadata["dynscm_profile"] == "benchmark_aligned_easy_plus_16k"
    assert metadata["dynscm_config"]["num_variables_min"] == 2
    assert metadata["dynscm_config"]["num_variables_max"] == 2
    assert metadata["dynscm_config"]["train_rows_min"] == 32
    assert metadata["dynscm_config"]["train_rows_max"] == 32
    assert metadata["dynscm_config"]["test_rows_min"] == 16
    assert metadata["dynscm_config"]["test_rows_max"] == 16
    assert metadata["dynscm_config"]["forecast_horizons"] == [1, 3, 6, 12]
    assert metadata["dynscm_config"]["num_regimes"] == 1
    assert metadata["dynscm_config"]["use_contemp_edges"] is False
    assert metadata["dynscm_config"]["max_lagged_parents"] == 2
    assert metadata["dynscm_config"]["mechanism_type_probs"] == [0.85, 0.15]
    assert metadata["dynscm_config"]["noise_family"] == "normal"
    assert metadata["dynscm_config"]["missing_mode"] == "off"
    assert metadata["dynscm_config"]["kernel_family_probs"] == [0.8, 0.2]


def test_dynscm_cli_benchmark_aligned_mechanism_profile_only_increases_mechanism_mix(
    tmp_path: Path,
):
    repo_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "dynscm_benchmark_aligned_mechanism_profile.h5"

    cmd = [
        sys.executable,
        "-m",
        "tfmplayground.priors",
        "--lib",
        "dynscm",
        "--dynscm_profile",
        "benchmark_aligned_mechanism_16k",
        "--num_batches",
        "1",
        "--batch_size",
        "1",
        "--max_classes",
        "0",
        "--np_seed",
        "34",
        "--dynscm_seed",
        "34",
        "--dynscm_workers",
        "1",
        "--dynscm_worker_blas_threads",
        "1",
        "--no_dynscm_compute_spectral_diagnostics",
        "--save_path",
        str(output_path),
    ]
    subprocess.run(
        cmd,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    with h5py.File(output_path, "r") as f:
        raw_metadata = f["dump_metadata_json"][()]
        metadata_text = (
            raw_metadata.decode("utf-8")
            if isinstance(raw_metadata, bytes)
            else str(raw_metadata)
        )
        metadata = json.loads(metadata_text)

    assert metadata["dynscm_profile"] == "benchmark_aligned_mechanism_16k"
    assert metadata["dynscm_config"]["num_regimes"] == 1
    assert metadata["dynscm_config"]["missing_mode"] == "off"
    assert metadata["dynscm_config"]["noise_family"] == "normal"
    assert metadata["dynscm_config"]["max_lagged_parents"] == 2
    assert metadata["dynscm_config"]["mechanism_type_probs"] == [0.7, 0.3]
    assert metadata["dynscm_config"]["kernel_family_probs"] == [0.8, 0.2]


def test_dynscm_cli_benchmark_aligned_edges_soft_profile_adds_only_small_edge_drift(
    tmp_path: Path,
):
    repo_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "dynscm_benchmark_aligned_edges_soft_profile.h5"

    cmd = [
        sys.executable,
        "-m",
        "tfmplayground.priors",
        "--lib",
        "dynscm",
        "--dynscm_profile",
        "benchmark_aligned_edges_soft_16k",
        "--num_batches",
        "1",
        "--batch_size",
        "1",
        "--max_classes",
        "0",
        "--np_seed",
        "36",
        "--dynscm_seed",
        "36",
        "--dynscm_workers",
        "1",
        "--dynscm_worker_blas_threads",
        "1",
        "--no_dynscm_compute_spectral_diagnostics",
        "--save_path",
        str(output_path),
    ]
    subprocess.run(
        cmd,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    with h5py.File(output_path, "r") as f:
        raw_metadata = f["dump_metadata_json"][()]
        metadata_text = (
            raw_metadata.decode("utf-8")
            if isinstance(raw_metadata, bytes)
            else str(raw_metadata)
        )
        metadata = json.loads(metadata_text)

    assert metadata["dynscm_profile"] == "benchmark_aligned_edges_soft_16k"
    assert metadata["dynscm_config"]["num_regimes"] == 1
    assert metadata["dynscm_config"]["missing_mode"] == "off"
    assert metadata["dynscm_config"]["noise_family"] == "normal"
    assert metadata["dynscm_config"]["max_lagged_parents"] == 2
    assert metadata["dynscm_config"]["mechanism_type_probs"] == [0.7, 0.3]
    assert metadata["dynscm_config"]["kernel_family_probs"] == [0.8, 0.2]
    assert metadata["dynscm_config"]["lagged_edge_add_prob"] == 0.005
    assert metadata["dynscm_config"]["lagged_edge_del_prob"] == 0.005


def test_dynscm_cli_benchmark_aligned_medium_graph_profile_isolates_structure_changes(
    tmp_path: Path,
):
    repo_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "dynscm_benchmark_aligned_medium_graph_profile.h5"

    cmd = [
        sys.executable,
        "-m",
        "tfmplayground.priors",
        "--lib",
        "dynscm",
        "--dynscm_profile",
        "benchmark_aligned_medium_graph_16k",
        "--num_batches",
        "1",
        "--batch_size",
        "1",
        "--max_classes",
        "0",
        "--np_seed",
        "35",
        "--dynscm_seed",
        "35",
        "--dynscm_workers",
        "1",
        "--dynscm_worker_blas_threads",
        "1",
        "--no_dynscm_compute_spectral_diagnostics",
        "--save_path",
        str(output_path),
    ]
    subprocess.run(
        cmd,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    with h5py.File(output_path, "r") as f:
        raw_metadata = f["dump_metadata_json"][()]
        metadata_text = (
            raw_metadata.decode("utf-8")
            if isinstance(raw_metadata, bytes)
            else str(raw_metadata)
        )
        metadata = json.loads(metadata_text)

    assert metadata["dynscm_profile"] == "benchmark_aligned_medium_graph_16k"
    assert metadata["dynscm_config"]["num_regimes"] == 1
    assert metadata["dynscm_config"]["missing_mode"] == "off"
    assert metadata["dynscm_config"]["noise_family"] == "normal"
    assert metadata["dynscm_config"]["max_lagged_parents"] == 2
    assert metadata["dynscm_config"]["mechanism_type_probs"] == [0.7, 0.3]
    assert metadata["dynscm_config"]["kernel_family_probs"] == [0.7, 0.3]


def test_dynscm_cli_benchmark_aligned_medium_noise_profile_adds_noise_diversity_only(
    tmp_path: Path,
):
    repo_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "dynscm_benchmark_aligned_medium_noise_profile.h5"

    cmd = [
        sys.executable,
        "-m",
        "tfmplayground.priors",
        "--lib",
        "dynscm",
        "--dynscm_profile",
        "benchmark_aligned_medium_noise_16k",
        "--num_batches",
        "1",
        "--batch_size",
        "1",
        "--max_classes",
        "0",
        "--np_seed",
        "39",
        "--dynscm_seed",
        "39",
        "--dynscm_workers",
        "1",
        "--dynscm_worker_blas_threads",
        "1",
        "--no_dynscm_compute_spectral_diagnostics",
        "--save_path",
        str(output_path),
    ]
    subprocess.run(
        cmd,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    with h5py.File(output_path, "r") as f:
        raw_metadata = f["dump_metadata_json"][()]
        metadata_text = (
            raw_metadata.decode("utf-8")
            if isinstance(raw_metadata, bytes)
            else str(raw_metadata)
        )
        metadata = json.loads(metadata_text)

    assert metadata["dynscm_profile"] == "benchmark_aligned_medium_noise_16k"
    assert metadata["dynscm_config"]["num_regimes"] == 1
    assert metadata["dynscm_config"]["missing_mode"] == "off"
    assert metadata["dynscm_config"]["noise_family_probs"] == [0.8, 0.2]
    assert metadata["dynscm_config"]["student_df_min"] == 4.0
    assert metadata["dynscm_config"]["student_df_max"] == 8.0
    assert metadata["dynscm_config"]["mechanism_type_probs"] == [0.7, 0.3]


def test_dynscm_cli_benchmark_aligned_medium_missing_profile_adds_missingness_last(
    tmp_path: Path,
):
    repo_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "dynscm_benchmark_aligned_medium_missing_profile.h5"

    cmd = [
        sys.executable,
        "-m",
        "tfmplayground.priors",
        "--lib",
        "dynscm",
        "--dynscm_profile",
        "benchmark_aligned_medium_missing_16k",
        "--num_batches",
        "1",
        "--batch_size",
        "1",
        "--max_classes",
        "0",
        "--np_seed",
        "43",
        "--dynscm_seed",
        "43",
        "--dynscm_workers",
        "1",
        "--dynscm_worker_blas_threads",
        "1",
        "--no_dynscm_compute_spectral_diagnostics",
        "--save_path",
        str(output_path),
    ]
    subprocess.run(
        cmd,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    with h5py.File(output_path, "r") as f:
        raw_metadata = f["dump_metadata_json"][()]
        metadata_text = (
            raw_metadata.decode("utf-8")
            if isinstance(raw_metadata, bytes)
            else str(raw_metadata)
        )
        metadata = json.loads(metadata_text)

    assert metadata["dynscm_profile"] == "benchmark_aligned_medium_missing_16k"
    assert metadata["dynscm_config"]["num_regimes"] == 1
    assert metadata["dynscm_config"]["noise_family_probs"] == [0.8, 0.2]
    assert metadata["dynscm_config"]["missing_mode_probs"] == [0.8, 0.2]
    assert metadata["dynscm_config"]["mechanism_type_probs"] == [0.7, 0.3]


def test_dynscm_cli_benchmark_aligned_medium_profile_is_moderately_diverse(
    tmp_path: Path,
):
    repo_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "dynscm_benchmark_aligned_medium_profile.h5"

    cmd = [
        sys.executable,
        "-m",
        "tfmplayground.priors",
        "--lib",
        "dynscm",
        "--dynscm_profile",
        "benchmark_aligned_medium_32k",
        "--num_batches",
        "1",
        "--batch_size",
        "1",
        "--max_classes",
        "0",
        "--np_seed",
        "37",
        "--dynscm_seed",
        "37",
        "--dynscm_workers",
        "1",
        "--dynscm_worker_blas_threads",
        "1",
        "--no_dynscm_compute_spectral_diagnostics",
        "--save_path",
        str(output_path),
    ]
    subprocess.run(
        cmd,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    with h5py.File(output_path, "r") as f:
        raw_metadata = f["dump_metadata_json"][()]
        metadata_text = (
            raw_metadata.decode("utf-8")
            if isinstance(raw_metadata, bytes)
            else str(raw_metadata)
        )
        metadata = json.loads(metadata_text)

    assert metadata["dynscm_profile"] == "benchmark_aligned_medium_32k"
    assert metadata["max_seq_len"] == 48
    assert metadata["max_features"] == 64
    assert metadata["dynscm_config"]["num_variables_min"] == 2
    assert metadata["dynscm_config"]["num_variables_max"] == 2
    assert metadata["dynscm_config"]["train_rows_min"] == 32
    assert metadata["dynscm_config"]["train_rows_max"] == 32
    assert metadata["dynscm_config"]["test_rows_min"] == 16
    assert metadata["dynscm_config"]["test_rows_max"] == 16
    assert metadata["dynscm_config"]["forecast_horizons"] == [1, 3, 6, 12]
    assert metadata["dynscm_config"]["explicit_lags"] == [0, 1, 2, 5, 10]
    assert metadata["dynscm_config"]["num_kernels"] == 3
    assert metadata["dynscm_config"]["num_regimes"] == 2
    assert metadata["dynscm_config"]["use_contemp_edges"] is False
    assert metadata["dynscm_config"]["mechanism_type_probs"] == [0.7, 0.3]
    assert metadata["dynscm_config"]["noise_family_probs"] == [0.8, 0.2]
    assert metadata["dynscm_config"]["missing_mode_probs"] == [0.8, 0.2]
    assert metadata["dynscm_config"]["kernel_family_probs"] == [0.7, 0.3]


def test_dynscm_cli_benchmark_aligned_full_profile_is_explicit_alias_of_rich_aligned(
    tmp_path: Path,
):
    repo_root = Path(__file__).resolve().parents[1]
    output_path = tmp_path / "dynscm_benchmark_aligned_full_profile.h5"

    cmd = [
        sys.executable,
        "-m",
        "tfmplayground.priors",
        "--lib",
        "dynscm",
        "--dynscm_profile",
        "benchmark_aligned_full_32k",
        "--num_batches",
        "1",
        "--batch_size",
        "1",
        "--max_classes",
        "0",
        "--np_seed",
        "41",
        "--dynscm_seed",
        "41",
        "--dynscm_workers",
        "1",
        "--dynscm_worker_blas_threads",
        "1",
        "--no_dynscm_compute_spectral_diagnostics",
        "--save_path",
        str(output_path),
    ]
    subprocess.run(
        cmd,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    with h5py.File(output_path, "r") as f:
        raw_metadata = f["dump_metadata_json"][()]
        metadata_text = (
            raw_metadata.decode("utf-8")
            if isinstance(raw_metadata, bytes)
            else str(raw_metadata)
        )
        metadata = json.loads(metadata_text)

    assert metadata["dynscm_profile"] == "benchmark_aligned_full_32k"
    assert metadata["max_seq_len"] == 48
    assert metadata["max_features"] == 64
    assert metadata["dynscm_config"]["num_variables_min"] == 2
    assert metadata["dynscm_config"]["num_variables_max"] == 2
    assert metadata["dynscm_config"]["train_rows_min"] == 32
    assert metadata["dynscm_config"]["train_rows_max"] == 32
    assert metadata["dynscm_config"]["test_rows_min"] == 16
    assert metadata["dynscm_config"]["test_rows_max"] == 16
    assert metadata["dynscm_config"]["forecast_horizons"] == [1, 3, 6, 12]
    assert metadata["dynscm_config"]["explicit_lags"] == [0, 1, 2, 5, 10]
    assert metadata["dynscm_config"]["num_kernels"] == 3
    assert metadata["dynscm_config"]["num_regimes"] == 5
    assert metadata["dynscm_config"]["shared_order"] is False
    assert metadata["dynscm_config"]["share_base_graph"] is False
    assert metadata["dynscm_config"]["mechanism_type_probs"] == [0.35, 0.65]
    assert metadata["dynscm_config"]["noise_family_probs"] == [0.4, 0.6]
