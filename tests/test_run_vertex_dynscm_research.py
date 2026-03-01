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


def test_run_vertex_dynscm_research_dry_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    _write_fake_gcloud(bin_dir)

    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    completed = subprocess.run(
        [
            "bash",
            "scripts/run_vertex_dynscm_research.sh",
            "--run-prefix",
            "research-test",
            "--dry-run",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    stdout = completed.stdout
    assert "[dry-run] profile=medium32k_live_baseline" in stdout
    assert "[dry-run] profile=medium32k_live_guardrails" in stdout
    assert "[dry-run] profile=medium32k_live_batch_homogeneous" in stdout
    assert "[dry-run] profile=medium32k_live_mode_ladder" in stdout
    assert "[dry-run] profile=medium32k_live_mixture" in stdout
    assert stdout.count("warm_start_checkpoint=") == 5
    assert "synthetic_eval=enabled" in stdout
