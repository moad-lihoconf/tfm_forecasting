from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_script_module(script_name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rank_mod = _load_script_module("eval_dynscm_target_stable_rank.py")


def test_phase1_ranker_reruns_top_two_and_writes_decision_schema(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def _fake_eval_run(
        *,
        profile_name: str,
        run_name: str,
        eval_steps: int,
        device,
        dynscm_workers: int,
        dynscm_worker_blas_threads: int,
    ):
        del profile_name, device, dynscm_workers, dynscm_worker_blas_threads
        fast_scores = {
            "run-a": (0.40, 1.00),
            "run-b": (0.30, 1.01),
            "run-c": (0.20, 1.02),
            "run-d": (0.10, 1.02),
        }
        full_scores = {
            "run-d": (0.08, 1.02),
            "run-c": (0.12, 1.02),
        }
        target_loss, stable_loss = (
            full_scores[run_name] if eval_steps == 64 else fast_scores[run_name]
        )
        return {
            "suites": {
                "target_eval": {
                    "loss": target_loss,
                    "skipped_fraction": 0.0,
                },
                "stable_eval": {
                    "loss": stable_loss,
                    "skipped_fraction": 0.0,
                },
            }
        }

    monkeypatch.setattr(rank_mod, "_evaluate_run", _fake_eval_run)
    ranking_path = tmp_path / "ranking.json"
    rank_mod.main(
        [
            "--device",
            "cpu",
            "--fast_eval_steps",
            "16",
            "--full_eval_steps",
            "64",
            "--full_top_k",
            "2",
            "--ranking_path",
            str(ranking_path),
            "--run",
            "profile-a",
            "run-a",
            "--run",
            "profile-b",
            "run-b",
            "--run",
            "profile-c",
            "run-c",
            "--run",
            "profile-d",
            "run-d",
        ]
    )

    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    assert [row["rank"] for row in ranking] == [1, 2, 3, 4]
    assert ranking[0]["run_name"] == "run-d"
    assert ranking[0]["decision"] == "promote"
    assert ranking[0]["eval_pass"] == "full"
    assert ranking[1]["run_name"] == "run-c"
    assert ranking[1]["decision"] == "promote"
    assert ranking[1]["eval_pass"] == "full"
    assert ranking[2]["decision"] == "hold"
    assert ranking[3]["decision"] == "hold"
