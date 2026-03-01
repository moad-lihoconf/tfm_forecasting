from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _write_fake_gcloud(bin_dir: Path) -> None:
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


def test_repeated_batch_overfit_sanity_local_dry_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    prior = tmp_path / "prior.h5"
    prior.write_text("prior", encoding="utf-8")

    cmd = [
        "bash",
        "scripts/run_repeated_batch_overfit_sanity.sh",
        "--dump",
        str(prior),
        "--run-name",
        "overfit-local",
        "--dry-run",
    ]
    completed = subprocess.run(
        cmd,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    stdout = completed.stdout
    assert "[overfit-sanity] mode=local" in stdout
    assert "poetry run python pretrain_regression.py" in stdout
    assert "--grad_clip_norm" in stdout


def test_repeated_batch_overfit_sanity_vertex_dry_run_uses_t4_defaults(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    _write_fake_gcloud(bin_dir)

    prior = tmp_path / "prior.h5"
    prior.write_text("prior", encoding="utf-8")

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    cmd = [
        "bash",
        "scripts/run_repeated_batch_overfit_sanity.sh",
        "--vertex",
        "--dump",
        str(prior),
        "--run-name",
        "overfit-vertex",
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
    assert "[overfit-sanity] mode=vertex" in stdout
    assert "acceleratorType: 'NVIDIA_TESLA_T4'" in stdout
    assert "--display-name tfm-regression-overfit-vertex" in stdout
