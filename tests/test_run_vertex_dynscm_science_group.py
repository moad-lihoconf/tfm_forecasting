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


def test_run_vertex_dynscm_science_group_dry_run_temporal(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    _write_fake_gcloud(bin_dir)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    completed = subprocess.run(
        [
            "bash",
            "scripts/run_vertex_dynscm_science_group.sh",
            "--group",
            "temporal_ablation",
            "--run-prefix",
            "science-test",
            "--dry-run",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    stdout = completed.stdout
    assert (
        "[dry-run] group=temporal_ablation profile=temporal_length_only_16k" in stdout
    )
    assert (
        "[dry-run] group=temporal_ablation profile=temporal_regimes_only_16k" in stdout
    )
    assert "[dry-run] group=temporal_ablation profile=temporal_drift_only_16k" in stdout
    assert (
        "[dry-run] group=temporal_ablation profile=temporal_regimes_plus_drift_16k"
        in stdout
    )
    assert (
        "[dry-run] group=temporal_ablation profile=temporal_length_plus_regimes_16k"
        in stdout
    )
    assert (
        "[dry-run] group=temporal_ablation profile=temporal_full_medium32k_reference"
        in stdout
    )
    assert stdout.count("learnability_audit=enabled") == 6
    assert stdout.count("benchmark_compare=enabled") == 6
    assert stdout.count("dynscm_workers=8") == 6
    assert stdout.count("dynscm_worker_blas_threads=1") == 6


def test_run_vertex_dynscm_science_group_dry_run_benchmark(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    _write_fake_gcloud(bin_dir)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    completed = subprocess.run(
        [
            "bash",
            "scripts/run_vertex_dynscm_science_group.sh",
            "--group",
            "benchmark_contract",
            "--run-prefix",
            "science-test",
            "--dry-run",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    stdout = completed.stdout
    assert (
        "[dry-run] group=benchmark_contract profile=benchmark_contract_observed_easy"
        in stdout
    )
    assert (
        "[dry-run] group=benchmark_contract "
        "profile=benchmark_contract_observed_temporal" in stdout
    )
    assert stdout.count("learnability_audit=enabled") == 2
    assert stdout.count("prior_audit=enabled") == 2


def test_run_vertex_dynscm_science_group_dry_run_normalization(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    _write_fake_gcloud(bin_dir)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    completed = subprocess.run(
        [
            "bash",
            "scripts/run_vertex_dynscm_science_group.sh",
            "--group",
            "normalization_ablation",
            "--run-prefix",
            "science-test",
            "--dry-run",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    stdout = completed.stdout
    assert (
        "[dry-run] group=normalization_ablation profile=mode_ladder_norm_none" in stdout
    )
    assert (
        "[dry-run] group=normalization_ablation profile=mode_ladder_norm_zscore"
        in stdout
    )
    assert (
        "[dry-run] group=normalization_ablation profile=mode_ladder_norm_clamped"
        in stdout
    )


def test_run_vertex_dynscm_science_group_dry_run_allows_worker_overrides(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    _write_fake_gcloud(bin_dir)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    completed = subprocess.run(
        [
            "bash",
            "scripts/run_vertex_dynscm_science_group.sh",
            "--group",
            "generator_learnability",
            "--run-prefix",
            "science-test",
            "--dynscm-workers",
            "6",
            "--dynscm-worker-blas-threads",
            "2",
            "--dry-run",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    stdout = completed.stdout
    assert stdout.count("dynscm_workers=6") == 3
    assert stdout.count("dynscm_worker_blas_threads=2") == 3
