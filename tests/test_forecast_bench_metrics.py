from __future__ import annotations

import numpy as np

from tfmplayground.benchmarks.forecasting.metrics import (
    bootstrap_ci,
    compute_proxy_metrics,
    compute_regression_metrics,
    mase,
    rmse,
    smape,
)


def test_regression_metrics_match_expected_toy_values():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 3.0, 2.0, 5.0])
    insample = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    assert np.isclose(rmse(y_true, y_pred), np.sqrt(0.75))
    assert smape(y_true, y_pred) > 0.0
    assert mase(y_true, y_pred, insample=insample, seasonality=1) > 0.0

    metrics = compute_regression_metrics(
        y_true, y_pred, insample=insample, seasonality=1
    )
    assert set(metrics) == {"rmse", "smape", "mase"}


def test_proxy_metrics_compute_balanced_accuracy_and_auc():
    y_true = np.array([0, 1, 2, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.1, 0.8, 0.1],
            [0.2, 0.5, 0.3],
            [0.2, 0.7, 0.1],
        ]
    )

    metrics = compute_proxy_metrics(y_true, y_pred, y_proba)
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0
    assert np.isfinite(metrics["macro_auroc"])


def test_bootstrap_ci_is_seed_deterministic():
    values = np.array([0.1, -0.2, 0.3, 0.4, -0.1], dtype=np.float64)
    ci_a = bootstrap_ci(values, n_bootstrap=500, confidence_level=0.95, seed=7)
    ci_b = bootstrap_ci(values, n_bootstrap=500, confidence_level=0.95, seed=7)
    assert np.allclose(ci_a, ci_b)
