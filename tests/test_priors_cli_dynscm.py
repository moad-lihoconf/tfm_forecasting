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
