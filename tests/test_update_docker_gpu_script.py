from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _write_fake_gcloud(bin_dir: Path) -> None:
    _write_executable(
        bin_dir / "gcloud",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [[ "$1" == "config" && "$2" == "get-value" ]]; then\n'
        '  case "$3" in\n'
        "    project) echo test-project ; exit 0 ;;\n"
        "    ai/region) echo europe-west4 ; exit 0 ;;\n"
        "  esac\n"
        "fi\n"
        "exit 1\n",
    )


def _write_fake_docker(bin_dir: Path, log_path: Path) -> None:
    _write_executable(
        bin_dir / "docker",
        f'#!/usr/bin/env bash\nset -euo pipefail\necho "$@" >> "{log_path}"\nexit 0\n',
    )


def test_update_docker_gpu_uses_latest_by_default(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    log_path = tmp_path / "docker.log"

    _write_fake_gcloud(bin_dir)
    _write_fake_docker(bin_dir, log_path)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    completed = subprocess.run(
        ["bash", "scripts/update_docker_gpu.sh", "--push"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    stdout = completed.stdout
    latest_image = (
        "europe-west4-docker.pkg.dev/test-project/tfm-forecasting/trainer-gpu:latest"
    )
    assert f"[gpu] building: {latest_image}" in stdout

    docker_calls = log_path.read_text(encoding="utf-8").splitlines()
    assert any("build" in call for call in docker_calls)
    assert any(call == f"push {latest_image}" for call in docker_calls)
    assert not any(call.startswith("tag ") for call in docker_calls)


def test_update_docker_gpu_custom_tag_also_updates_latest(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    log_path = tmp_path / "docker.log"

    _write_fake_gcloud(bin_dir)
    _write_fake_docker(bin_dir, log_path)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    completed = subprocess.run(
        [
            "bash",
            "scripts/update_docker_gpu.sh",
            "--push",
            "--tag",
            "manual-tag",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    stdout = completed.stdout
    manual_image = (
        "europe-west4-docker.pkg.dev/test-project/tfm-forecasting/"
        "trainer-gpu:manual-tag"
    )
    latest_image = (
        "europe-west4-docker.pkg.dev/test-project/tfm-forecasting/trainer-gpu:latest"
    )
    assert f"[gpu] building: {manual_image}" in stdout
    assert f"[gpu] tagging: {latest_image}" in stdout
    assert f"[gpu] pushing: {latest_image}" in stdout

    docker_calls = log_path.read_text(encoding="utf-8").splitlines()
    assert any(call == f"tag {manual_image} {latest_image}" for call in docker_calls)
    assert any(call == f"push {latest_image}" for call in docker_calls)


def test_update_docker_gpu_rejects_missing_option_value(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    log_path = tmp_path / "docker.log"

    _write_fake_gcloud(bin_dir)
    _write_fake_docker(bin_dir, log_path)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"

    completed = subprocess.run(
        ["bash", "scripts/update_docker_gpu.sh", "--tag", "--push"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 2
    assert "Error: --tag requires a non-empty value." in completed.stderr
    assert not log_path.exists()
