from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch

from tfmplayground.benchmarks.forecasting.adapters import (
    AdapterSkipError,
    NanoTabPFNForecastAdapter,
    NICLClientAdapter,
    NICLRegressionAdapter,
    build_forecast_table_from_series,
    default_regression_adapters,
)
from tfmplayground.benchmarks.forecasting.config import ForecastBenchmarkConfig
from tfmplayground.interface import (
    _OFFICIAL_REGRESSOR_ARCH,
    NanoTabPFNRegressor,
    init_model_from_state_dict_file,
)
from tfmplayground.model import NanoTabPFNModel


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


def _make_raw_official_checkpoint(path: Path) -> Path:
    model = NanoTabPFNModel(**_OFFICIAL_REGRESSOR_ARCH)
    torch.save(model.state_dict(), path)
    return path


def _make_bucket_file(path: Path) -> Path:
    torch.save(torch.linspace(-1.0, 1.0, 101), path)
    return path


def test_init_model_from_state_dict_file_supports_raw_state_dict(tmp_path: Path):
    raw_path = _make_raw_official_checkpoint(tmp_path / "raw_model.pth")

    model = init_model_from_state_dict_file(raw_path)

    assert isinstance(model, NanoTabPFNModel)
    assert model.decoder.linear2.weight.shape[0] == 100


def test_nanotabpfn_regressor_uses_default_raw_checkpoint_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    _make_raw_official_checkpoint(checkpoints_dir / "nanotabpfn_regressor.pth")
    _make_bucket_file(checkpoints_dir / "nanotabpfn_regressor_buckets.pth")
    monkeypatch.chdir(tmp_path)

    regressor = NanoTabPFNRegressor(model=None, dist=None, device="cpu")

    assert isinstance(regressor.model, NanoTabPFNModel)


def test_nanotabpfn_regressor_supports_scalar_regression_without_dist():
    model = NanoTabPFNModel(
        num_attention_heads=1,
        embedding_size=8,
        mlp_hidden_size=16,
        num_layers=1,
        num_outputs=1,
        dropout=0.0,
    )

    def _forward(data, single_eval_pos: int, num_mem_chunks: int = 8):
        del num_mem_chunks
        x, _y = data
        return torch.zeros(
            (x.shape[0], x.shape[1] - int(single_eval_pos), 1),
            dtype=torch.float32,
            device=x.device,
        )

    model.forward = _forward  # type: ignore[method-assign]
    regressor = NanoTabPFNRegressor(
        model=model,
        dist=None,
        device="cpu",
    )
    x_train = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    y_train = np.array([10.0, 12.0, 14.0], dtype=np.float32)
    x_test = np.array([[3.0], [4.0]], dtype=np.float32)

    regressor.fit(x_train, y_train)
    preds = regressor.predict(x_test)

    assert preds.shape == (2,)
    assert np.allclose(preds, np.mean(y_train))


def _make_wrapped_checkpoint(
    path: Path,
    *,
    target_normalization: str | None = None,
    target_std_floor: float | None = None,
    training_nested: bool = True,
) -> Path:
    model = NanoTabPFNModel(
        num_attention_heads=1,
        embedding_size=8,
        mlp_hidden_size=16,
        num_layers=1,
        num_outputs=1,
        dropout=0.0,
    )
    payload: dict[str, object] = {
        "architecture": {
            "num_attention_heads": 1,
            "embedding_size": 8,
            "mlp_hidden_size": 16,
            "num_layers": 1,
            "num_outputs": 1,
            "dropout": 0.0,
        },
        "model": model.state_dict(),
    }
    if target_normalization is not None:
        if training_nested:
            payload["training"] = {
                "target_normalization": target_normalization,
                "target_std_floor": target_std_floor,
            }
        else:
            payload["target_normalization"] = target_normalization
            payload["target_std_floor"] = target_std_floor
    torch.save(payload, path)
    return path


def test_nanotabpfn_regressor_reads_target_normalization_none_from_ckpt(
    tmp_path: Path,
):
    ckpt_path = _make_wrapped_checkpoint(
        tmp_path / "ckpt_none_top_level.pth",
        target_normalization="none",
        target_std_floor=0.123,
        training_nested=False,
    )
    regressor = NanoTabPFNRegressor(model=str(ckpt_path), dist=None, device="cpu")

    assert regressor.target_normalization == "none"

    def _forward(data, single_eval_pos: int, num_mem_chunks: int = 8):
        del num_mem_chunks
        x, _y = data
        return torch.zeros(
            (x.shape[0], x.shape[1] - int(single_eval_pos), 1),
            dtype=torch.float32,
            device=x.device,
        )

    regressor.model.forward = _forward  # type: ignore[method-assign]
    x_train = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    y_train = np.array([10.0, 12.0, 14.0], dtype=np.float32)
    x_test = np.array([[3.0], [4.0]], dtype=np.float32)
    regressor.fit(x_train, y_train)

    preds = regressor.predict(x_test)

    # With target_normalization=none, predict() should not renormalize.
    assert np.allclose(regressor.y_train_n, y_train)
    assert np.allclose(preds, 0.0)


def test_nanotabpfn_regressor_reads_target_normalization_clamped_and_applies_floor(
    tmp_path: Path,
):
    ckpt_path = _make_wrapped_checkpoint(
        tmp_path / "ckpt_clamped_nested.pth",
        target_normalization="per_function_clamped",
        target_std_floor=0.5,
        training_nested=True,
    )
    regressor = NanoTabPFNRegressor(model=str(ckpt_path), dist=None, device="cpu")

    assert regressor.target_normalization == "per_function_clamped"
    assert regressor.target_std_floor == 0.5

    x_train = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    y_train = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    regressor.fit(x_train, y_train)

    assert regressor.y_train_std >= 0.5
    assert np.allclose(regressor.y_train_n, 0.0)


def test_init_model_from_state_dict_file_rejects_invalid_raw_checkpoint(tmp_path: Path):
    bad_path = tmp_path / "bad_model.pth"
    torch.save(OrderedDict({"decoder.linear1.weight": torch.zeros((2, 2))}), bad_path)

    with pytest.raises(ValueError, match="Unsupported checkpoint format"):
        init_model_from_state_dict_file(bad_path)


def test_default_regression_adapters_honors_enabled_model_selection():
    cfg = ForecastBenchmarkConfig.from_dict(
        {
            "models": {
                "enabled_regression_models": [
                    "nanotabpfn_standard",
                    "nanotabpfn_dynscm",
                    "nicl_regression",
                ],
                "nicl_regression_mode": "quantized_proxy",
                "nicl_regression_endpoint": "https://example.com/reg",
            }
        }
    )

    adapters = default_regression_adapters(cfg, device="cpu")

    assert set(adapters) == {
        "nanotabpfn_standard",
        "nanotabpfn_dynscm",
        "nicl_regression",
    }
