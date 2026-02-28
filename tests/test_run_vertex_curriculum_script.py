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
        "    ai/region) echo europe-west4 ; exit 0 ;;\n"
        "    ai/staging_bucket) echo gs://test-bucket ; exit 0 ;;\n"
        "  esac\n"
        "fi\n"
        'echo fake-gcloud-called "$@"\n'
        "exit 0\n",
        encoding="utf-8",
    )
    gcloud.chmod(0o755)


def test_run_vertex_curriculum_dry_run_prints_stage_plan(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    _write_fake_gcloud(bin_dir)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            "scripts/run_vertex_curriculum.sh",
            "--run-prefix",
            "curriculum-test",
            "--from-stage",
            "benchmark_aligned_easy_16k",
            "--to-stage",
            "benchmark_aligned_mechanism_16k",
            "--dry-run",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    stdout = completed.stdout
    assert "[dry-run] stage=benchmark_aligned_easy_16k" in stdout
    assert "run_name=curriculum-test-easy-16k" in stdout
    assert (
        "prior_uri=gs://test-bucket/tfm_forecasting/priors/"
        "benchmark_aligned_easy_16k.h5"
    ) in stdout
    assert "epochs=10" in stdout
    assert "early_stopping_patience=3" in stdout
    assert "early_stopping_min_delta=1e-4" in stdout
    assert "warm_start_checkpoint=<none>" in stdout

    assert "[dry-run] stage=benchmark_aligned_easy_plus_16k" in stdout
    assert "epochs=20" in stdout
    assert "early_stopping_patience=5" in stdout
    assert "early_stopping_min_delta=1e-5" in stdout
    assert (
        "warm_start_checkpoint=gs://test-bucket/tfm_forecasting/runs/"
        "curriculum-test-easy-16k/checkpoints/best_checkpoint.pth"
    ) in stdout

    assert "[dry-run] stage=benchmark_aligned_mechanism_16k" in stdout
    assert "run_name=curriculum-test-mechanism-16k" in stdout
    assert stdout.count("early_stopping_patience=10") == 1
    assert stdout.count("early_stopping_min_delta=1e-5") == 2
    assert "stream_logs=enabled" in stdout
    assert (
        "summary_file=workdir/curriculum/curriculum-test/curriculum_summary.tsv"
    ) in stdout
