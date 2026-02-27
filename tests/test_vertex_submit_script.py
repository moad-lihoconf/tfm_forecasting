from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _write_fake_gcloud(bin_dir: Path) -> Path:
    gcloud = bin_dir / "gcloud"
    gcloud.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [[ "$1" == "config" && "$2" == "get-value" ]]; then\n'
        '  case "$3" in\n'
        "    project) echo test-project ; exit 0 ;;\n"
        "    ai/region) echo us-central1 ; exit 0 ;;\n"
        "    ai/staging_bucket) echo gs://test-bucket ; exit 0 ;;\n"
        "  esac\n"
        "fi\n"
        'echo fake-gcloud-called "$@"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    gcloud.chmod(0o755)
    return gcloud


def test_submit_vertex_script_dry_run_local_prior(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    _write_fake_gcloud(bin_dir)

    local_prior = tmp_path / "prior.h5"
    local_prior.write_text("prior", encoding="utf-8")

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    cmd = [
        "bash",
        "scripts/submit_vertex_regression.sh",
        "--priordump",
        str(local_prior),
        "--run-name",
        "run_local",
        "--dry-run",
    ]
    completed = subprocess.run(
        cmd,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    stdout = completed.stdout
    expected_prior_uri = (
        "[dry-run] prior_uri: gs://test-bucket/tfm_forecasting/priors/prior.h5"
    )
    expected_image_uri = (
        "[dry-run] image: "
        "us-central1-docker.pkg.dev/test-project/tfm-forecasting/trainer-gpu:latest"
    )
    expected_accelerator = "acceleratorType: 'NVIDIA_L4'"
    assert expected_prior_uri in stdout
    assert "gcloud storage cp" in stdout
    assert "--display-name tfm-regression-run_local" in stdout
    assert expected_image_uri in stdout
    assert expected_accelerator in stdout


def test_submit_vertex_script_dry_run_gcs_prior(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    _write_fake_gcloud(bin_dir)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    cmd = [
        "bash",
        "scripts/submit_vertex_regression.sh",
        "--priordump",
        "gs://existing-bucket/tfm_forecasting/priors/already.h5",
        "--run-name",
        "run_gcs",
        "--dry-run",
    ]
    completed = subprocess.run(
        cmd,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    stdout = completed.stdout
    expected_prior_uri = (
        "[dry-run] prior_uri: gs://existing-bucket/tfm_forecasting/priors/already.h5"
    )
    expected_image_uri = (
        "[dry-run] image: "
        "us-central1-docker.pkg.dev/test-project/tfm-forecasting/trainer-gpu:latest"
    )
    expected_accelerator = "acceleratorType: 'NVIDIA_L4'"
    assert expected_prior_uri in stdout
    assert "upload command: <none, gs:// priordump already accessible>" in stdout
    assert "gcloud storage cp" not in stdout
    assert expected_image_uri in stdout
    assert expected_accelerator in stdout
