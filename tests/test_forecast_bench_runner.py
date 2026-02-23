from __future__ import annotations

from pathlib import Path

import numpy as np

from tfmplayground.benchmarks.forecasting.config import ForecastBenchmarkConfig
from tfmplayground.benchmarks.forecasting.datasets import DatasetBundle
from tfmplayground.benchmarks.forecasting.runner import (
    evaluate_regression,
    run_benchmark,
    summarize_regression,
)


class _MeanAdapter:
    def fit_predict(self, x_train, y_train, x_test):
        return np.full((x_test.shape[0],), float(np.mean(y_train)), dtype=np.float64)


class _LastLagAdapter:
    def fit_predict(self, x_train, y_train, x_test):
        # First feature typically contains lag-0 value in current feature layout.
        return np.asarray(x_test[:, 0], dtype=np.float64)


def _make_cfg(tmp_path: Path) -> ForecastBenchmarkConfig:
    return ForecastBenchmarkConfig.from_dict(
        {
            "mode": "regression",
            "output_dir": str(tmp_path / "out"),
            "datasets": {
                "dataset_names": ["dummy"],
                "allow_download": False,
                "cache_dir": str(tmp_path / "cache"),
            },
            "protocol": {
                "horizons": [1, 2, 3],
                "context_rows": 12,
                "test_rows": 6,
                "max_feature_lag": 8,
                "explicit_lags": [0, 1, 2, 4],
                "num_kernels": 1,
            },
            "stats": {
                "bootstrap_samples": 400,
                "confidence_level": 0.9,
                "claim_metrics": ["mase", "smape", "rmse"],
                "min_metrics_to_pass": 2,
            },
        }
    )


def test_evaluate_regression_outputs_finite_rows_and_is_deterministic(tmp_path: Path):
    cfg = _make_cfg(tmp_path)
    rng = np.random.default_rng(11)

    series = np.stack(
        [
            np.linspace(0.0, 1.0, 220),
            np.sin(np.linspace(0.0, 12.0, 220)) + 0.1 * rng.normal(size=220),
        ],
        axis=0,
    )
    suite = {
        "dummy": DatasetBundle(
            name="dummy",
            series=series,
            frequency="daily",
            seasonality=7,
            skipped=False,
            skip_reason=None,
        )
    }
    adapters = {
        "nanotabpfn_standard": _MeanAdapter(),
        "nanotabpfn_dynscm": _LastLagAdapter(),
        "tabicl_regressor": _MeanAdapter(),
    }

    rows_a = evaluate_regression(cfg, suite=suite, adapters=adapters)
    rows_b = evaluate_regression(cfg, suite=suite, adapters=adapters)

    ok_a = rows_a.loc[rows_a["status"] == "ok"]
    assert not ok_a.empty
    assert np.isfinite(ok_a[["rmse", "smape", "mase"]].to_numpy()).all()

    assert rows_a.equals(rows_b)


def test_summarize_regression_computes_claim_fields(tmp_path: Path):
    cfg = _make_cfg(tmp_path)

    rows = np.array(
        [
            ("d", 0, 1, "nanotabpfn_standard", 1.0, 1.0, 1.0),
            ("d", 0, 1, "nanotabpfn_dynscm", 0.5, 0.5, 0.5),
            ("d", 0, 1, "tabicl_regressor", 1.2, 1.2, 1.2),
            ("d", 1, 3, "nanotabpfn_standard", 1.0, 1.0, 1.0),
            ("d", 1, 3, "nanotabpfn_dynscm", 0.4, 0.4, 0.4),
            ("d", 1, 3, "tabicl_regressor", 1.3, 1.3, 1.3),
        ],
        dtype=[
            ("dataset", "U10"),
            ("series_id", "i4"),
            ("horizon", "i4"),
            ("model", "U32"),
            ("rmse", "f8"),
            ("smape", "f8"),
            ("mase", "f8"),
        ],
    )

    import pandas as pd

    df = pd.DataFrame(rows)
    df["status"] = "ok"
    df["skip_reason"] = ""

    summary = summarize_regression(df, cfg)
    assert "comparisons" in summary
    assert "claim" in summary
    assert "primary_claim_pass" in summary["claim"]


def test_run_benchmark_writes_artifacts(monkeypatch, tmp_path: Path):
    cfg = _make_cfg(tmp_path)

    suite = {
        "dummy": DatasetBundle(
            name="dummy",
            series=np.linspace(0.0, 1.0, 220)[None, :],
            frequency="daily",
            seasonality=7,
            skipped=False,
            skip_reason=None,
        )
    }

    import tfmplayground.benchmarks.forecasting.runner as runner_mod

    monkeypatch.setattr(runner_mod, "load_suite", lambda _cfg: suite)
    monkeypatch.setattr(
        runner_mod,
        "default_regression_adapters",
        lambda _cfg, device="cpu": {
            "nanotabpfn_standard": _MeanAdapter(),
            "nanotabpfn_dynscm": _LastLagAdapter(),
            "tabicl_regressor": _MeanAdapter(),
        },
    )

    artifacts = run_benchmark(cfg, device="cpu")
    assert artifacts.regression_rows_path is not None
    assert artifacts.regression_summary_path is not None
    assert artifacts.report_path.exists()
    assert artifacts.regression_rows_path.exists()
    assert artifacts.regression_summary_path.exists()
