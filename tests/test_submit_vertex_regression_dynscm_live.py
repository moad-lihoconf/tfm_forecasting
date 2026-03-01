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


def test_submit_vertex_live_script_dry_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    _write_fake_gcloud(bin_dir)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    completed = subprocess.run(
        [
            "bash",
            "scripts/submit_vertex_regression_dynscm_live.sh",
            "--research-profile",
            "medium32k_live_baseline",
            "--run-name",
            "run_live",
            "--dry-run",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    stdout = completed.stdout
    assert "[dry-run] research_profile: medium32k_live_baseline" in stdout
    assert "/app/pretrain_regression_dynscm_live.py" in stdout
    assert "--research_profile=medium32k_live_baseline" in stdout
    assert "--format=value(name)" in stdout
