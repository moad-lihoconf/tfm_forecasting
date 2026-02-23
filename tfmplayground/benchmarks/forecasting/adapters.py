"""Model adapters and featurization wrappers for forecasting benchmarks."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import requests

if TYPE_CHECKING:
    import torch

from tfmplayground.interface import NanoTabPFNClassifier, NanoTabPFNRegressor
from tfmplayground.priors.dynscm.config import DynSCMConfig
from tfmplayground.priors.dynscm.features import build_forecasting_table

from .config import ForecastBenchmarkConfig

_CRITICAL_EXCEPTIONS = (KeyboardInterrupt, SystemExit, MemoryError)

__all__ = [
    "AdapterSkipError",
    "ForecastTable",
    "build_forecast_table_from_series",
    "default_proxy_adapters",
    "default_regression_adapters",
]


class AdapterSkipError(RuntimeError):
    """Raised when an adapter cannot run and should be skipped."""


@dataclass(frozen=True, slots=True)
class ForecastTable:
    """Forecast table with split metadata."""

    x: np.ndarray  # (N, F)
    y: np.ndarray  # (N,)
    t_idx: np.ndarray  # (N,)
    h_idx: np.ndarray  # (N,)
    split_index: int


def build_forecast_table_from_series(
    cfg: ForecastBenchmarkConfig,
    series_1d: np.ndarray,
    *,
    t_idx: np.ndarray,
    h_idx: np.ndarray,
    split_index: int,
    seed: int,
) -> ForecastTable:
    """Build one forecast table from a univariate raw series using DynSCM featurization."""
    values = np.asarray(series_1d, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("series_1d must be a 1D array.")
    if values.size <= int(np.max(h_idx)):
        raise ValueError("series_1d is too short for requested horizons.")

    observed = np.isfinite(values)
    series_filled = np.where(observed, values, 0.0)

    # DynSCM shape config enforces >=2 variables. For univariate benchmark
    # series we duplicate the target as an auxiliary channel.
    series_2d = np.stack([series_filled, series_filled], axis=1)
    obs_2d = np.stack([observed, observed], axis=1)

    series_3d = series_2d[None, :, :]
    obs_mask = obs_2d[None, :, :]

    dyn_cfg = _make_featurization_cfg(cfg, series_length=values.size)
    x_3d, y_2d, meta = build_forecasting_table(
        dyn_cfg,
        series_3d,
        np.array([0], dtype=np.int64),
        n_train=split_index,
        n_test=t_idx.size - split_index,
        t_idx=t_idx[None, :],
        h_idx=h_idx[None, :],
        obs_mask=obs_mask,
        seed=seed,
    )

    x = x_3d[0]
    y = y_2d[0]
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError("Non-finite values encountered in featurized table.")

    # Hard no-leak check by construction.
    if np.any(
        meta["t_idx"][0] < max(dyn_cfg.max_feature_lag, max(dyn_cfg.explicit_lags))
    ):
        raise RuntimeError("No-leak guarantee violated: origin below lag floor.")

    return ForecastTable(
        x=x.astype(np.float64, copy=False),
        y=y.astype(np.float64, copy=False),
        t_idx=np.asarray(meta["t_idx"][0], dtype=np.int64),
        h_idx=np.asarray(meta["h_idx"][0], dtype=np.int64),
        split_index=int(split_index),
    )


def _make_featurization_cfg(
    cfg: ForecastBenchmarkConfig, *, series_length: int
) -> DynSCMConfig:
    payload = {
        "num_variables_min": 2,
        "num_variables_max": 2,
        "series_length_min": int(series_length),
        "series_length_max": int(series_length),
        "max_lag": max(
            int(cfg.protocol.max_feature_lag), int(max(cfg.protocol.explicit_lags))
        ),
        "forecast_horizons": tuple(int(h) for h in cfg.protocol.horizons),
        "train_rows_min": int(cfg.protocol.context_rows),
        "train_rows_max": int(cfg.protocol.context_rows),
        "test_rows_min": int(cfg.protocol.test_rows),
        "test_rows_max": int(cfg.protocol.test_rows),
        "max_feature_lag": int(cfg.protocol.max_feature_lag),
        "explicit_lags": tuple(int(v) for v in cfg.protocol.explicit_lags),
        "num_kernels": int(cfg.protocol.num_kernels),
        "add_mask_channels": bool(cfg.protocol.add_mask_channels),
        "random_seed": int(cfg.seed),
    }
    return DynSCMConfig.from_dict(payload)


class NanoTabPFNForecastAdapter:
    """Regression adapter backed by NanoTabPFNRegressor."""

    def __init__(
        self,
        *,
        name: str,
        model_path: str | None,
        dist_path: str | None,
        device: str | torch.device,
        num_mem_chunks: int = 8,
        model_factory: Callable[..., Any] | None = None,
    ):
        self.name = name
        self.model_path = model_path
        self.dist_path = dist_path
        self.device = device
        self.num_mem_chunks = int(num_mem_chunks)
        self._model_factory = model_factory or NanoTabPFNRegressor

    def fit_predict(
        self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
    ) -> np.ndarray:
        try:
            model = self._model_factory(
                model=self.model_path,
                dist=self.dist_path,
                device=self.device,
                num_mem_chunks=self.num_mem_chunks,
            )
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            pred_arr = np.asarray(pred, dtype=np.float64)
            if pred_arr.shape != (x_test.shape[0],):
                raise ValueError(
                    f"Unexpected prediction shape {pred_arr.shape}, expected {(x_test.shape[0],)}"
                )
            return pred_arr
        except _CRITICAL_EXCEPTIONS:
            raise
        except Exception as exc:  # pragma: no cover
            raise AdapterSkipError(f"{self.name} failed: {exc}") from exc


class TabICLForecastAdapter:
    """Regression adapter backed by TabICLRegressor."""

    def __init__(
        self,
        *,
        checkpoint_version: str,
        model_path: str | None,
        device: str | torch.device,
        estimator_factory: Callable[..., Any] | None = None,
    ):
        self.name = "tabicl_regressor"
        self.checkpoint_version = checkpoint_version
        self.model_path = model_path
        self.device = device
        self._estimator_factory = estimator_factory

    def fit_predict(
        self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
    ) -> np.ndarray:
        try:
            if self._estimator_factory is None:
                from tabicl.sklearn.regressor import TabICLRegressor

                estimator_factory = TabICLRegressor
            else:
                estimator_factory = self._estimator_factory

            model = estimator_factory(
                checkpoint_version=self.checkpoint_version,
                model_path=self.model_path,
                device=self.device,
            )
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
            pred_arr = np.asarray(pred, dtype=np.float64)
            if pred_arr.shape != (x_test.shape[0],):
                raise ValueError(
                    f"Unexpected prediction shape {pred_arr.shape}, expected {(x_test.shape[0],)}"
                )
            return pred_arr
        except _CRITICAL_EXCEPTIONS:
            raise
        except Exception as exc:  # pragma: no cover
            raise AdapterSkipError(f"{self.name} failed: {exc}") from exc


class NanoTabPFNClassifierAdapter:
    """Classification adapter backed by NanoTabPFNClassifier."""

    def __init__(
        self,
        *,
        name: str,
        model_path: str | None,
        device: str | torch.device,
        num_mem_chunks: int = 8,
        model_factory: Callable[..., Any] | None = None,
    ):
        self.name = name
        self.model_path = model_path
        self.device = device
        self.num_mem_chunks = int(num_mem_chunks)
        self._model_factory = model_factory or NanoTabPFNClassifier

    def fit_predict_proba(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            model = self._model_factory(
                model=self.model_path,
                device=self.device,
                num_mem_chunks=self.num_mem_chunks,
            )
            model.fit(x_train, y_train)
            pred = np.asarray(model.predict(x_test), dtype=np.int64)
            proba = np.asarray(model.predict_proba(x_test), dtype=np.float64)
            return pred, proba
        except _CRITICAL_EXCEPTIONS:
            raise
        except Exception as exc:  # pragma: no cover
            raise AdapterSkipError(f"{self.name} failed: {exc}") from exc


class TabICLClassifierAdapter:
    """Classification adapter backed by TabICLClassifier."""

    def __init__(
        self,
        *,
        checkpoint_version: str,
        model_path: str | None,
        device: str | torch.device,
        estimator_factory: Callable[..., Any] | None = None,
    ):
        self.name = "tabicl_classifier"
        self.checkpoint_version = checkpoint_version
        self.model_path = model_path
        self.device = device
        self._estimator_factory = estimator_factory

    def fit_predict_proba(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            if self._estimator_factory is None:
                from tabicl.sklearn.classifier import TabICLClassifier

                estimator_factory = TabICLClassifier
            else:
                estimator_factory = self._estimator_factory

            model = estimator_factory(
                checkpoint_version=self.checkpoint_version,
                model_path=self.model_path,
                device=self.device,
            )
            model.fit(x_train, y_train)
            pred = np.asarray(model.predict(x_test), dtype=np.int64)
            proba = np.asarray(model.predict_proba(x_test), dtype=np.float64)
            return pred, proba
        except _CRITICAL_EXCEPTIONS:
            raise
        except Exception as exc:  # pragma: no cover
            raise AdapterSkipError(f"{self.name} failed: {exc}") from exc


class NICLClientAdapter:
    """HTTP adapter for NICL API predictions on proxy classification track."""

    def __init__(
        self,
        *,
        api_url: str,
        timeout_seconds: float,
        max_retries: int,
        token_env: str = "NICL_API_TOKEN",
        session: requests.Session | None = None,
    ):
        self.name = "nicl_api"
        self.api_url = api_url
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.token_env = token_env
        self.session = session or requests.Session()

    def fit_predict_proba(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        *,
        num_classes: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        token = os.getenv(self.token_env)
        if not token:
            raise AdapterSkipError(f"Missing NICL token in env var {self.token_env}.")

        payload = {
            "task": "classification",
            "x_train": np.asarray(x_train, dtype=np.float64).tolist(),
            "y_train": np.asarray(y_train, dtype=np.int64).tolist(),
            "x_test": np.asarray(x_test, dtype=np.float64).tolist(),
            "num_classes": int(num_classes),
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                result = response.json()
                pred, proba = _parse_nicl_response(result)
                if (
                    pred.shape[0] != x_test.shape[0]
                    or proba.shape[0] != x_test.shape[0]
                ):
                    raise ValueError("NICL response length does not match test size.")
                return pred, proba
            except Exception as exc:  # pragma: no cover
                last_error = exc
                if attempt + 1 < self.max_retries:
                    time.sleep(0.5 * (2**attempt))

        raise AdapterSkipError(f"NICL request failed after retries: {last_error}")


def _parse_nicl_response(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    proba_raw = next(
        (
            payload[k]
            for k in ("probabilities", "y_proba", "predictions_proba")
            if k in payload
        ),
        None,
    )
    pred_raw = next(
        (payload[k] for k in ("predictions", "y_pred") if k in payload),
        None,
    )

    if proba_raw is None:
        raise ValueError("NICL response missing probabilities field.")

    proba = np.asarray(proba_raw, dtype=np.float64)
    if proba.ndim == 1:
        proba = proba[:, None]

    if pred_raw is None:
        pred = np.argmax(proba, axis=1)
    else:
        pred = np.asarray(pred_raw, dtype=np.int64)

    return pred, proba


class UnavailableRegressionAdapter:
    """Placeholder adapter that always skips."""

    def __init__(self, name: str, reason: str):
        self.name = name
        self.reason = reason

    def fit_predict(
        self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
    ) -> np.ndarray:
        raise AdapterSkipError(f"{self.name} unavailable: {self.reason}")


def default_regression_adapters(
    cfg: ForecastBenchmarkConfig,
    *,
    device: str | torch.device,
) -> dict[str, Any]:
    """Build default regression adapters used in the benchmark."""
    adapters: dict[str, Any] = {
        "nanotabpfn_standard": NanoTabPFNForecastAdapter(
            name="nanotabpfn_standard",
            model_path=cfg.models.model_standard_ckpt,
            dist_path=cfg.models.model_standard_dist,
            device=device,
        ),
        "tabicl_regressor": TabICLForecastAdapter(
            checkpoint_version=cfg.models.tabicl_checkpoint_version,
            model_path=cfg.models.tabicl_model_path,
            device=device,
        ),
    }

    if cfg.models.model_dynscm_ckpt is None and cfg.models.model_dynscm_dist is None:
        adapters["nanotabpfn_dynscm"] = UnavailableRegressionAdapter(
            "nanotabpfn_dynscm",
            "Provide --model_dynscm_ckpt (and optional --model_dynscm_dist).",
        )
    else:
        adapters["nanotabpfn_dynscm"] = NanoTabPFNForecastAdapter(
            name="nanotabpfn_dynscm",
            model_path=cfg.models.model_dynscm_ckpt,
            dist_path=cfg.models.model_dynscm_dist,
            device=device,
        )

    return adapters


def default_proxy_adapters(
    cfg: ForecastBenchmarkConfig,
    *,
    device: str | torch.device,
) -> dict[str, Any]:
    """Build default proxy-classification adapters."""
    return {
        "nanotabpfn_classifier": NanoTabPFNClassifierAdapter(
            name="nanotabpfn_classifier",
            model_path=None,
            device=device,
        ),
        "tabicl_classifier": TabICLClassifierAdapter(
            checkpoint_version=cfg.models.tabicl_checkpoint_version,
            model_path=cfg.models.tabicl_model_path,
            device=device,
        ),
        "nicl_api": NICLClientAdapter(
            api_url=cfg.models.nicl_api_url,
            timeout_seconds=cfg.models.nicl_timeout_seconds,
            max_retries=cfg.models.nicl_max_retries,
        ),
    }
