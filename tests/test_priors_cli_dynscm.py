"""CLI roundtrip test for DynSCM prior dump integration."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

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

    prior = PriorDumpDataLoader(
        filename=str(output_path),
        num_steps=1,
        batch_size=2,
        device=torch.device("cpu"),
    )
    batch = next(iter(prior))

    assert set(batch) == {"x", "y", "target_y", "single_eval_pos"}
    assert batch["x"].shape[0] == 2
    assert batch["y"].shape[0] == 2
    assert batch["target_y"].shape[0] == 2

    split = int(batch["single_eval_pos"])
    assert 0 < split < batch["x"].shape[1]
    assert torch.isfinite(batch["x"]).all()
    assert torch.isfinite(batch["y"]).all()
    assert not np.isnan(batch["y"][:, :split].numpy()).any()
