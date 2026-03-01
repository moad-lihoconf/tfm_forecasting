from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


def _load_script_module(script_name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


compare_live_mod = _load_script_module(
    "compare_live_dynscm_profile_to_forecast_benchmark.py"
)


class _FakeLoader:
    def __init__(self, batch: dict[str, torch.Tensor]):
        self._batch = batch

    def __iter__(self):
        yield self._batch

    def close(self) -> None:
        return None


def test_summarize_live_source_and_main_write_expected_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    profile = compare_live_mod.get_research_profile("benchmark_contract_observed_easy")

    x = np.zeros((2, 48, 64), dtype=np.float32)
    x[:, :48, :37] = 1.0
    y = np.zeros((2, 48), dtype=np.float32)
    y[0, :32] = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    y[0, 32:48] = np.linspace(0.5, 2.0, 16, dtype=np.float32)
    y[1, :32] = np.linspace(1.0, 2.0, 32, dtype=np.float32)
    y[1, 32:48] = np.linspace(2.0, 4.0, 16, dtype=np.float32)
    batch = {
        "x": torch.from_numpy(x),
        "y": torch.from_numpy(y),
        "sampled_n_train": torch.tensor([32, 32]),
        "sampled_n_test": torch.tensor([16, 16]),
        "sampled_num_vars": torch.tensor([2, 2]),
        "sampled_pre_budget_feature_count": torch.tensor([37, 37]),
        "sampled_mechanism_type_id": torch.tensor([0, 0]),
        "sampled_noise_family_id": torch.tensor([0, 0]),
        "sampled_missing_mode_id": torch.tensor([0, 0]),
        "sampled_kernel_family_id": torch.tensor([0, 0]),
    }

    monkeypatch.setattr(
        compare_live_mod.live_train,
        "_build_prior_loader",
        lambda **kwargs: _FakeLoader(batch),
    )
    benchmark_summary = {
        "num_variables": {"support": [2], "distribution": {"2": 1.0}, "stats": {}},
        "context_rows": {
            "support": [32],
            "distribution": {"32": 1.0},
            "stats": {"min": 32.0, "mean": 32.0, "median": 32.0, "max": 32.0},
        },
        "test_rows": {
            "support": [16],
            "distribution": {"16": 1.0},
            "stats": {"min": 16.0, "mean": 16.0, "median": 16.0, "max": 16.0},
        },
        "horizons": {"support": [1, 3, 6, 12], "distribution": {}},
        "feature_count_before_padding": {
            "support": [37],
            "stats": {"min": 37.0, "mean": 37.0, "median": 37.0, "max": 37.0},
        },
        "active_feature_count": {
            "support": [37],
            "stats": {"min": 37.0, "mean": 37.0, "median": 37.0, "max": 37.0},
        },
        "lag_set": [0, 1, 2, 5, 10],
        "num_kernels": 3,
        "mask_channels": True,
        "target_row_count": {
            "support": [16],
            "distribution": {"16": 1.0},
            "stats": {"min": 16.0, "mean": 16.0, "median": 16.0, "max": 16.0},
        },
        "train_target_std": {"min": 0.1, "mean": 0.2, "median": 0.2, "max": 0.3},
        "test_target_std": {"min": 0.1, "mean": 0.3, "median": 0.3, "max": 0.5},
        "low_variance_target_fraction": 0.0,
    }
    benchmark_samples = {
        "num_variables": np.asarray([2, 2], dtype=np.int64),
        "context_rows": np.asarray([32, 32], dtype=np.int64),
        "test_rows": np.asarray([16, 16], dtype=np.int64),
        "horizons": np.asarray([1, 3, 6, 12], dtype=np.int64),
        "feature_count_before_padding": np.asarray([37, 37], dtype=np.int64),
        "active_feature_count": np.asarray([37, 37], dtype=np.int64),
        "target_row_count": np.asarray([16, 16], dtype=np.int64),
        "train_target_std": np.asarray([0.2, 0.3], dtype=np.float64),
        "test_target_std": np.asarray([0.3, 0.5], dtype=np.float64),
    }
    monkeypatch.setattr(
        compare_live_mod,
        "summarize_benchmark",
        lambda *args, **kwargs: (benchmark_summary, benchmark_samples),
    )

    summary, samples = compare_live_mod.summarize_live_source(
        profile=profile,
        source_name="train",
        sample_steps=1,
        batch_size=2,
        device=torch.device("cpu"),
        workers=1,
        worker_blas_threads=1,
    )
    assert summary["feature_count_before_padding"]["support"] == [37]
    assert summary["active_feature_count"]["support"] == [37]
    assert summary["missing_mode_distribution"] == {"off": 1.0}
    assert samples["num_variables"].tolist() == [2, 2]

    json_out = tmp_path / "compare.json"
    markdown_out = tmp_path / "compare.md"
    audit_out = tmp_path / "prior_audit.json"
    compare_live_mod.main(
        [
            "--research_profile",
            "benchmark_contract_observed_easy",
            "--source",
            "train",
            "--sample_steps",
            "1",
            "--batchsize",
            "2",
            "--dynscm_workers",
            "1",
            "--json-out",
            str(json_out),
            "--markdown-out",
            str(markdown_out),
            "--prior-audit-json",
            str(audit_out),
        ]
    )

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    audit_payload = json.loads(audit_out.read_text(encoding="utf-8"))
    assert payload["research_profile"] == "benchmark_contract_observed_easy"
    assert payload["prior_summary"]["feature_count_before_padding"]["support"] == [37]
    assert isinstance(payload["mismatches"], list)
    assert audit_payload["mask_channels"] is False
    assert "# DynSCM Prior vs Forecast Benchmark" in markdown_out.read_text(
        encoding="utf-8"
    )
