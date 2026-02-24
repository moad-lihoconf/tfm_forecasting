from __future__ import annotations

import numpy as np
import pytest

from tfmplayground.benchmarks.forecasting.adapters import (
    AdapterSkipError,
    NanoTabPFNForecastAdapter,
    NICLClientAdapter,
    NICLRegressionAdapter,
    build_forecast_table_from_series,
    default_regression_adapters,
)
from tfmplayground.benchmarks.forecasting.config import ForecastBenchmarkConfig


class _DummyRegressor:
    def __init__(self, **kwargs):
        pass

    def fit(self, x_train, y_train):
        self.mean_ = float(np.mean(y_train))

    def predict(self, x_test):
        return np.full((x_test.shape[0],), self.mean_, dtype=np.float64)


def test_build_forecast_table_from_series_no_leak_and_finite():
    cfg = ForecastBenchmarkConfig()
    series = np.linspace(0.0, 1.0, 256)
    t_idx = np.arange(64, 64 + cfg.protocol.context_rows + cfg.protocol.test_rows)
    h_idx = np.ones_like(t_idx)

    table = build_forecast_table_from_series(
        cfg,
        series,
        t_idx=t_idx,
        h_idx=h_idx,
        split_index=cfg.protocol.context_rows,
        seed=42,
    )

    assert table.x.shape[0] == cfg.protocol.context_rows + cfg.protocol.test_rows
    assert np.isfinite(table.x).all()
    assert np.isfinite(table.y).all()
    assert np.all(
        table.t_idx[: table.split_index] <= table.t_idx[table.split_index :].min()
    )


def test_nanotabpfn_adapter_with_injected_model_factory():
    adapter = NanoTabPFNForecastAdapter(
        name="nano_dummy",
        model_path=None,
        dist_path=None,
        device="cpu",
        model_factory=_DummyRegressor,
    )

    x_train = np.random.default_rng(0).normal(size=(8, 4))
    y_train = np.random.default_rng(1).normal(size=(8,))
    x_test = np.random.default_rng(2).normal(size=(3, 4))

    pred = adapter.fit_predict(x_train, y_train, x_test)
    assert pred.shape == (3,)
    assert np.isfinite(pred).all()


def test_default_regression_adapters_marks_dynscm_as_unavailable_when_missing_ckpt():
    cfg = ForecastBenchmarkConfig()
    adapters = default_regression_adapters(cfg, device="cpu")

    assert "nanotabpfn_dynscm" in adapters
    with pytest.raises(AdapterSkipError, match="unavailable"):
        adapters["nanotabpfn_dynscm"].fit_predict(
            np.zeros((2, 2)), np.zeros((2,)), np.zeros((1, 2))
        )


def test_nicl_adapter_requires_token(monkeypatch):
    monkeypatch.delenv("NEURALK_API_KEY", raising=False)
    monkeypatch.delenv("NICL_API_TOKEN", raising=False)
    adapter = NICLClientAdapter(
        api_url="https://example.com",
        timeout_seconds=1.0,
        max_retries=1,
    )

    with pytest.raises(AdapterSkipError, match="Missing NICL token"):
        adapter.fit_predict_proba(
            np.zeros((4, 3)),
            np.zeros((4,), dtype=np.int64),
            np.zeros((2, 3)),
            num_classes=2,
        )


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def post(self, *args, **kwargs):
        return _FakeResponse(
            {
                "probabilities": [[0.8, 0.2], [0.3, 0.7]],
                "predictions": [0, 1],
            }
        )


def test_nicl_adapter_parses_response(monkeypatch):
    monkeypatch.setenv("NEURALK_API_KEY", "secret")
    adapter = NICLClientAdapter(
        api_url="https://example.com",
        timeout_seconds=1.0,
        max_retries=1,
        session=_FakeSession(),
    )

    pred, proba = adapter.fit_predict_proba(
        np.zeros((4, 3)),
        np.zeros((4,), dtype=np.int64),
        np.zeros((2, 3)),
        num_classes=2,
    )
    assert pred.shape == (2,)
    assert proba.shape == (2, 2)


class _CaptureSession:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def post(self, *args, **kwargs):
        self.requests.append(kwargs.get("json", {}))
        return _FakeResponse(self.payload)


def test_nicl_regression_native_parses_float_predictions(monkeypatch):
    monkeypatch.setenv("NEURALK_API_KEY", "secret")
    session = _CaptureSession({"predictions": [1.5, 2.5, 3.5]})
    adapter = NICLRegressionAdapter(
        api_url="https://example.com/reg",
        timeout_seconds=1.0,
        max_retries=1,
        mode="native",
        session=session,
    )
    pred = adapter.fit_predict(
        np.zeros((4, 3), dtype=np.float64),
        np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        np.zeros((3, 3), dtype=np.float64),
    )
    assert pred.shape == (3,)
    assert np.allclose(pred, [1.5, 2.5, 3.5])
    assert session.requests[0]["task"] == "regression"


def test_nicl_regression_quantized_proxy_uses_train_only_binning(monkeypatch):
    monkeypatch.setenv("NEURALK_API_KEY", "secret")
    session = _CaptureSession(
        {
            "probabilities": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "predictions": [0, 3],
        }
    )
    adapter = NICLRegressionAdapter(
        api_url="https://example.com/cls",
        timeout_seconds=1.0,
        max_retries=1,
        mode="quantized_proxy",
        session=session,
        proxy_num_classes="auto",
        min_samples_per_class=1,
    )
    y_train = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    pred = adapter.fit_predict(
        np.zeros((6, 3), dtype=np.float64),
        y_train,
        np.zeros((2, 3), dtype=np.float64),
    )
    assert pred.shape == (2,)
    assert np.isfinite(pred).all()
    sent = session.requests[0]
    assert sent["task"] == "classification"
    assert "y_train" in sent
    assert sent["num_classes"] >= 2
    assert sent["num_classes"] <= len(y_train)
    # no y_test leakage should ever be sent
    assert "y_test" not in sent
    # quantized classes should be integer encoded
    assert all(isinstance(v, int) for v in sent["y_train"])


def test_nicl_regression_rejects_invalid_proxy_num_classes_string():
    with pytest.raises(ValueError, match="proxy_num_classes"):
        NICLRegressionAdapter(
            api_url="https://example.com/cls",
            timeout_seconds=1.0,
            max_retries=1,
            mode="quantized_proxy",
            proxy_num_classes="bad",
        )
