"""Smoke test for DynSCM generation benchmark script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_dynscm_benchmark_script_outputs_metrics_json():
    repo_root = Path(__file__).resolve().parents[1]
    cmd = [
        sys.executable,
        "scripts/benchmark_dynscm_generation.py",
        "--num_batches",
        "1",
        "--batch_size",
        "2",
        "--max_seq_len",
        "16",
        "--max_features",
        "16",
        "--workers",
        "1",
        "--worker_blas_threads",
        "1",
        "--seed",
        "23",
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
    ]
    completed = subprocess.run(
        cmd,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["num_batches"] == 1
    assert payload["batch_size"] == 2
    assert payload["workers"] == 1
    assert payload["worker_blas_threads"] == 1
    assert payload["sec_per_batch"] > 0.0
    assert payload["ms_per_sample"] > 0.0
