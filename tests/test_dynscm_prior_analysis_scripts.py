from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from tfmplayground.benchmarks.forecasting.adapters import ForecastTable
from tfmplayground.benchmarks.forecasting.config import ForecastBenchmarkConfig
from tfmplayground.benchmarks.forecasting.datasets import DatasetBundle
from tfmplayground.benchmarks.forecasting.protocol import RollingOriginIndices


def _load_script_module(script_name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


compare_mod = _load_script_module("compare_dynscm_prior_to_forecast_benchmark.py")
profile_mod = _load_script_module("profile_dynscm_prior_diversity.py")
validate_mod = _load_script_module("validate_dynscm_generation_invariants.py")


def _write_prior_fixture(path: Path) -> None:
    x = np.zeros((4, 48, 64), dtype=np.float32)
    x[:2, :48, :45] = 1.0
    x[2:, :48, :55] = 2.0
    y = np.zeros((4, 48), dtype=np.float32)
    y[0, :48] = np.linspace(0.0, 1.0, 48, dtype=np.float32)
    y[1, :48] = y[0, :48]
    y[2, :48] = np.linspace(1.0, 2.0, 48, dtype=np.float32)
    y[3, :48] = np.linspace(2.0, 3.0, 48, dtype=np.float32)
    num_datapoints = np.full((4,), 48, dtype=np.int32)
    single_eval_pos = np.full((4,), 32, dtype=np.int32)
    sampled_num_vars = np.array([2, 2, 2, 2], dtype=np.int32)
    sampled_n_train = np.full((4,), 32, dtype=np.int32)
    sampled_n_test = np.full((4,), 16, dtype=np.int32)
    pre_budget_feature_count = np.array([45, 45, 55, 55], dtype=np.int32)
    metadata_payload = {
        "dynscm_family_id_mappings": {
            "mechanism_type": {"0": "linear_var", "1": "linear_plus_residual"},
            "noise_family": {"0": "normal", "1": "student_t"},
            "missing_mode": {
                "0": "off",
                "1": "mcar",
                "2": "mar",
                "3": "mnar_lite",
                "4": "mix",
            },
            "kernel_family": {"0": "exp_decay", "1": "power_law", "2": "mix"},
        },
        "dynscm_config": {
            "forecast_horizons": [1, 3, 6, 12],
            "explicit_lags": [0, 1, 2, 5, 10],
            "num_kernels": 3,
            "max_feature_lag": 32,
            "add_mask_channels": True,
            "add_time_feature": True,
            "add_horizon_feature": True,
            "add_log_horizon": True,
            "add_seasonality": True,
        },
    }

    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=x)
        f.create_dataset("y", data=y)
        f.create_dataset("num_datapoints", data=num_datapoints)
        f.create_dataset("single_eval_pos", data=single_eval_pos)
        f.create_dataset("sampled_num_vars", data=sampled_num_vars)
        f.create_dataset("sampled_n_train", data=sampled_n_train)
        f.create_dataset("sampled_n_test", data=sampled_n_test)
        f.create_dataset(
            "sampled_pre_budget_feature_count", data=pre_budget_feature_count
        )
        f.create_dataset(
            "sampled_mechanism_type_id", data=np.array([0, 1, 0, 1], dtype=np.int32)
        )
        f.create_dataset(
            "sampled_noise_family_id", data=np.array([0, 1, 0, 1], dtype=np.int32)
        )
        f.create_dataset(
            "sampled_missing_mode_id", data=np.array([0, 1, 2, 4], dtype=np.int32)
        )
        f.create_dataset(
            "sampled_kernel_family_id", data=np.array([0, 1, 2, 2], dtype=np.int32)
        )
        f.create_dataset("problem_type", data="regression", dtype=h5py.string_dtype())
        f.create_dataset(
            "dump_metadata_json",
            data=json.dumps(metadata_payload),
            dtype=h5py.string_dtype(),
        )


def test_profile_dynscm_prior_diversity_reports_duplicate_metrics(
    tmp_path: Path,
) -> None:
    prior_path = tmp_path / "prior_fixture.h5"
    _write_prior_fixture(prior_path)

    report = profile_mod.build_diversity_report(str(prior_path), sample_limit=None)

    assert report["num_tables"] == 4
    assert report["inspected_tables"] == 4
    assert report["duplicate_fraction"] > 0.0
    assert report["near_duplicate_fraction"] >= report["duplicate_fraction"]
    assert report["feature_truncation_fraction"] == 0.0
    assert report["feature_budget_saturation_fraction"] == 0.0
    assert report["feature_block_counts"]["total_pre_budget_theoretical"]["max"] == 37.0
    assert report["status"] == "warning"


def test_compare_prior_to_benchmark_reports_expected_mismatches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prior_path = tmp_path / "prior_fixture.h5"
    _write_prior_fixture(prior_path)

    cfg = ForecastBenchmarkConfig.from_dict(
        {
            "datasets": {
                "dataset_names": ["exchange_rate"],
                "allow_download": False,
                "max_series_per_dataset": 1,
            }
        }
    )
    fake_suite = {
        "exchange_rate": DatasetBundle(
            name="exchange_rate",
            series=np.asarray([np.linspace(0.0, 1.0, 80, dtype=np.float64)]),
            frequency="daily",
            seasonality=7,
            skipped=False,
            skip_reason=None,
        )
    }

    monkeypatch.setattr(compare_mod, "load_suite", lambda _cfg: fake_suite)
    monkeypatch.setattr(
        compare_mod,
        "generate_rolling_origin_indices",
        lambda **kwargs: RollingOriginIndices(
            t_idx=np.arange(32, 80),
            h_idx=np.array([1] * 32 + [12] * 16, dtype=np.int64),
            n_train=32,
            n_test=16,
        ),
    )
    monkeypatch.setattr(
        compare_mod,
        "build_forecast_table_from_series",
        lambda *args, **kwargs: ForecastTable(
            x=np.ones((48, 37), dtype=np.float64),
            y=np.linspace(0.0, 1.0, 48, dtype=np.float64),
            t_idx=np.arange(32, 80),
            h_idx=np.array([1] * 32 + [12] * 16, dtype=np.int64),
            split_index=32,
        ),
    )

    prior_summary, prior_samples = compare_mod.summarize_prior_dump(
        str(prior_path), sample_limit=None
    )
    benchmark_summary, benchmark_samples = compare_mod.summarize_benchmark(
        cfg,
        dataset_limit=1,
        series_limit=1,
    )
    mismatches = compare_mod.build_mismatch_report(
        prior_summary,
        prior_samples,
        benchmark_summary,
        benchmark_samples,
    )

    assert (
        prior_summary["feature_block_counts"]["total_pre_budget_theoretical"]["max"]
        == 37.0
    )
    assert benchmark_summary["num_variables"]["support"] == [2]
    assert mismatches[0]["dimension"] in {
        "active_feature_count",
        "feature_count_before_padding",
    }
    assert any(
        mismatch["dimension"] == "num_variables" and mismatch["score"] == 0.0
        for mismatch in mismatches
    )


def test_validate_generation_invariants_passes_for_benchmark_profile() -> None:
    report = validate_mod.run_invariant_audit(
        dynscm_profile="benchmark_aligned_16k",
        num_samples=4,
        row_budget=48,
        feature_budget=64,
        seed=7,
    )

    assert report["status"] == "pass"
    invariants = report["invariants"]
    assert invariants["label_correctness"]["status"] == "pass"
    assert invariants["chronology_correctness"]["status"] == "pass"
    assert invariants["feature_budget_semantics"]["status"] == "pass"
    assert invariants["feature_block_ordering"]["status"] == "pass"
    assert invariants["missingness_handling"]["status"] == "pass"
    assert invariants["sampling_support"]["status"] == "pass"
    assert invariants["sampling_support"]["num_vars_support"] == [2]
    assert invariants["sampling_support"]["n_train_support"] == [32]


def test_validate_generation_invariants_passes_for_easy_plus_profile() -> None:
    report = validate_mod.run_invariant_audit(
        dynscm_profile="benchmark_aligned_easy_plus_16k",
        num_samples=4,
        row_budget=48,
        feature_budget=64,
        seed=11,
    )

    assert report["status"] == "pass"
    invariants = report["invariants"]
    assert invariants["label_correctness"]["status"] == "pass"
    assert invariants["chronology_correctness"]["status"] == "pass"
    assert invariants["feature_budget_semantics"]["status"] == "pass"
    assert invariants["feature_block_ordering"]["status"] == "pass"
    assert invariants["missingness_handling"]["status"] == "pass"
    assert invariants["sampling_support"]["status"] == "pass"
    assert invariants["sampling_support"]["num_vars_support"] == [2]
    assert invariants["sampling_support"]["n_train_support"] == [32]
    assert invariants["sampling_support"]["n_test_support"] == [16]


def test_validate_generation_invariants_passes_for_easy_benchmark_profile() -> None:
    report = validate_mod.run_invariant_audit(
        dynscm_profile="benchmark_aligned_easy_16k",
        num_samples=4,
        row_budget=48,
        feature_budget=64,
        seed=11,
    )

    assert report["status"] == "pass"
    invariants = report["invariants"]
    assert invariants["label_correctness"]["status"] == "pass"
    assert invariants["chronology_correctness"]["status"] == "pass"
    assert invariants["sampling_support"]["status"] == "pass"
    assert invariants["sampling_support"]["num_vars_support"] == [2]
    assert invariants["sampling_support"]["n_train_support"] == [32]
    assert invariants["sampling_support"]["n_test_support"] == [16]
    assert invariants["sampling_support"]["horizon_support"] == [1, 3, 6, 12]
    assert invariants["sampling_support"]["n_test_support"] == [16]
    assert invariants["sampling_support"]["horizon_support"] == [1, 3, 6, 12]
