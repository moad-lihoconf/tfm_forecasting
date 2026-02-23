from __future__ import annotations

import numpy as np

from tfmplayground.benchmarks.forecasting.config import ForecastBenchmarkConfig
from tfmplayground.benchmarks.forecasting.datasets import DatasetBundle
from tfmplayground.benchmarks.forecasting.proxy_classification import (
    fit_quantile_binner,
    transform_to_classes,
)
from tfmplayground.benchmarks.forecasting.runner import evaluate_proxy, summarize_proxy


class _DummyClassifierAdapter:
    def fit_predict_proba(self, x_train, y_train, x_test, **kwargs):
        num_classes = int(np.max(y_train)) + 1
        pred = np.zeros((x_test.shape[0],), dtype=np.int64)
        proba = np.zeros((x_test.shape[0], num_classes), dtype=np.float64)
        proba[:, 0] = 1.0
        return pred, proba


def _make_cfg() -> ForecastBenchmarkConfig:
    return ForecastBenchmarkConfig.from_dict(
        {
            "mode": "proxy",
            "protocol": {
                "horizons": [1, 2],
                "context_rows": 12,
                "test_rows": 6,
                "max_feature_lag": 8,
                "explicit_lags": [0, 1, 2],
                "num_kernels": 0,
            },
            "proxy": {
                "num_classes": 3,
                "min_samples_per_class": 2,
            },
        }
    )


def test_quantile_binner_uses_train_distribution_only():
    y_train = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    edges = fit_quantile_binner(y_train, num_classes=3, min_samples_per_class=1)
    y_test = np.array([-10.0, 0.5, 3.5, 99.0])

    labels = transform_to_classes(y_test, edges)
    assert labels.shape == (4,)
    assert labels.min() >= 0
    assert labels.max() <= 2


def test_evaluate_proxy_and_summary_with_dummy_adapters():
    cfg = _make_cfg()
    suite = {
        "dummy": DatasetBundle(
            name="dummy",
            series=np.sin(np.linspace(0.0, 20.0, 240))[None, :],
            frequency="daily",
            seasonality=7,
            skipped=False,
            skip_reason=None,
        )
    }
    adapters = {
        "nanotabpfn_classifier": _DummyClassifierAdapter(),
        "tabicl_classifier": _DummyClassifierAdapter(),
        "nicl_api": _DummyClassifierAdapter(),
    }

    rows = evaluate_proxy(cfg, suite=suite, adapters=adapters)
    ok_rows = rows.loc[rows["status"] == "ok"]
    assert not ok_rows.empty
    assert np.isfinite(ok_rows[["balanced_accuracy"]].to_numpy()).all()

    summary = summarize_proxy(rows)
    assert "models" in summary
    assert len(summary["models"]) >= 1
