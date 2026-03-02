"""Model adapters and featurization wrappers for forecasting benchmarks."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import requests
import torch

from tfmplayground.interface import NanoTabPFNClassifier, NanoTabPFNRegressor
from tfmplayground.priors.dynscm.config import DynSCMConfig
from tfmplayground.priors.dynscm.features import build_forecasting_table

from .config import ForecastBenchmarkConfig
from .proxy_classification import (
    choose_num_classes,
    fit_quantile_binner,
    transform_to_classes,
)

_CRITICAL_EXCEPTIONS = (KeyboardInterrupt, SystemExit, MemoryError)

__all__ = [
    "AdapterSkipError",
    "ForecastTable",
    "build_forecast_table_from_series",
    "default_proxy_adapters",
    "default_regression_adapters",
    "NICLRegressionAdapter",
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
    """Build one forecast table from a univariate raw series."""
    values = np.asarray(series_1d, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("series_1d must be a 1D array.")
    if values.size <= np.max(h_idx):
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
        split_index=split_index,
    )


def _make_featurization_cfg(
    cfg: ForecastBenchmarkConfig, *, series_length: int
) -> DynSCMConfig:
    payload = {
        "num_variables_min": 2,
        "num_variables_max": 2,
        "series_length_min": series_length,
        "series_length_max": series_length,
        "max_lag": max(cfg.protocol.max_feature_lag, max(cfg.protocol.explicit_lags)),
        "forecast_horizons": cfg.protocol.horizons,
        "train_rows_min": cfg.protocol.context_rows,
        "train_rows_max": cfg.protocol.context_rows,
        "test_rows_min": cfg.protocol.test_rows,
        "test_rows_max": cfg.protocol.test_rows,
        "max_feature_lag": cfg.protocol.max_feature_lag,
        "explicit_lags": cfg.protocol.explicit_lags,
        "num_kernels": cfg.protocol.num_kernels,
        "add_mask_channels": cfg.protocol.add_mask_channels,
        "random_seed": cfg.seed,
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
        self.num_mem_chunks = num_mem_chunks
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
                    f"Unexpected prediction shape {pred_arr.shape}, "
                    f"expected {(x_test.shape[0],)}"
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
                    f"Unexpected prediction shape {pred_arr.shape}, "
                    f"expected {(x_test.shape[0],)}"
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
        self.num_mem_chunks = num_mem_chunks
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
        token_env: str = "NEURALK_API_KEY",
        session: requests.Session | None = None,
    ):
        self.name = "nicl_api"
        self.api_url = api_url
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
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
        token = _resolve_nicl_token(self.token_env)

        payload = {
            "task": "classification",
            "x_train": np.asarray(x_train, dtype=np.float64).tolist(),
            "y_train": np.asarray(y_train, dtype=np.int64).tolist(),
            "x_test": np.asarray(x_test, dtype=np.float64).tolist(),
            "num_classes": num_classes,
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
                pred, proba = _parse_nicl_classification_response(result)
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


class NICLRegressionAdapter:
    """NICL adapter for regression (`native`) or quantized proxy mode."""

    def __init__(
        self,
        *,
        api_url: str,
        timeout_seconds: float,
        max_retries: int,
        mode: str,
        token_env: str = "NEURALK_API_KEY",
        model_name: str = "nicl-small",
        proxy_num_classes: int | str = "auto",
        min_samples_per_class: int = 2,
        session: requests.Session | None = None,
    ):
        if mode not in {"native", "quantized_proxy"}:
            raise ValueError(f"Unsupported NICL regression mode: {mode!r}")
        self.name = "nicl_regression"
        self.api_url = api_url
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.mode = mode
        self.token_env = token_env
        self.model_name = model_name
        if isinstance(proxy_num_classes, str) and proxy_num_classes != "auto":
            raise ValueError("proxy_num_classes must be an int or the string 'auto'.")
        self.proxy_num_classes = proxy_num_classes
        self.min_samples_per_class = min_samples_per_class
        self.session = session or requests.Session()

    def fit_predict(
        self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
    ) -> np.ndarray:
        if self.mode == "native":
            return self._fit_predict_native(x_train, y_train, x_test)
        return self._fit_predict_quantized_proxy(x_train, y_train, x_test)

    def _fit_predict_native(
        self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
    ) -> np.ndarray:
        token = _resolve_nicl_token(self.token_env)
        payload = {
            "task": "regression",
            "model": self.model_name,
            "x_train": np.asarray(x_train, dtype=np.float64).tolist(),
            "y_train": np.asarray(y_train, dtype=np.float64).tolist(),
            "x_test": np.asarray(x_test, dtype=np.float64).tolist(),
        }
        result = _nicl_post_json(
            session=self.session,
            url=self.api_url,
            payload=payload,
            token=token,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
        )
        pred = _parse_nicl_regression_response(result)
        if pred.shape != (x_test.shape[0],):
            raise AdapterSkipError(
                "NICL regression response shape mismatch: "
                f"pred={pred.shape} expected={(x_test.shape[0],)}"
            )
        return pred

    def _fit_predict_quantized_proxy(
        self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray
    ) -> np.ndarray:
        token = _resolve_nicl_token(self.token_env)
        y_train_arr = np.asarray(y_train, dtype=np.float64)

        try:
            requested = (
                self.proxy_num_classes
                if self.proxy_num_classes == "auto"
                else self.proxy_num_classes
            )
            selected_num_classes = choose_num_classes(
                y_train_arr,
                num_classes=requested,  # type: ignore[arg-type]
                min_samples_per_class=self.min_samples_per_class,
            )
            edges = fit_quantile_binner(
                y_train_arr,
                num_classes=selected_num_classes,
                min_samples_per_class=self.min_samples_per_class,
            )
            y_train_cls = transform_to_classes(y_train_arr, edges)
        except Exception as exc:
            raise AdapterSkipError(
                f"NICL quantized proxy binning failed: {exc}"
            ) from exc

        num_classes = edges.size - 1
        payload = {
            "task": "classification",
            "model": self.model_name,
            "x_train": np.asarray(x_train, dtype=np.float64).tolist(),
            "y_train": np.asarray(y_train_cls, dtype=np.int64).tolist(),
            "x_test": np.asarray(x_test, dtype=np.float64).tolist(),
            "num_classes": num_classes,
        }
        result = _nicl_post_json(
            session=self.session,
            url=self.api_url,
            payload=payload,
            token=token,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
        )
        pred_cls, proba = _parse_nicl_classification_response(result)

        centers = _compute_train_bin_centers(y_train_arr, y_train_cls, num_classes)
        pred = _classification_outputs_to_regression(
            pred_cls=pred_cls,
            proba=proba,
            centers=centers,
        )
        if pred.shape != (x_test.shape[0],):
            raise AdapterSkipError(
                "NICL quantized proxy response shape mismatch: "
                f"pred={pred.shape} expected={(x_test.shape[0],)}"
            )
        return pred


def _parse_nicl_classification_response(
    payload: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
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


def _parse_nicl_regression_response(payload: dict[str, Any]) -> np.ndarray:
    raw = next(
        (
            payload[k]
            for k in ("predictions", "y_pred", "values", "prediction")
            if k in payload
        ),
        None,
    )
    if raw is None:
        raise AdapterSkipError("NICL regression response missing predictions field.")
    pred = np.asarray(raw, dtype=np.float64).reshape(-1)
    if not np.isfinite(pred).all():
        raise AdapterSkipError("NICL regression predictions contain non-finite values.")
    return pred


def _nicl_post_json(
    *,
    session: requests.Session,
    url: str,
    payload: dict[str, Any],
    token: str,
    timeout_seconds: float,
    max_retries: int,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = session.post(
                url,
                json=payload,
                headers=headers,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError("NICL response must be a JSON object.")
            return result
        except requests.HTTPError as exc:  # pragma: no cover
            last_error = _rewrite_nicl_endpoint_error(exc)
            if attempt + 1 < max_retries:
                time.sleep(0.5 * (2**attempt))
        except Exception as exc:  # pragma: no cover
            last_error = exc
            if attempt + 1 < max_retries:
                time.sleep(0.5 * (2**attempt))
    raise AdapterSkipError(f"NICL request failed after retries: {last_error}")


def _rewrite_nicl_endpoint_error(exc: requests.HTTPError) -> Exception:
    """Attach endpoint hint for the common dashboard-vs-API URL mismatch."""
    response = exc.response
    if response is None:
        return exc
    status_code = response.status_code
    request_url = str(response.url)
    if (
        status_code == 405
        and request_url.rstrip("/") == "https://prediction.neuralk-ai.com/predict"
    ):
        return RuntimeError(
            "405 Not Allowed at dashboard URL "
            "'https://prediction.neuralk-ai.com/predict'. "
            "Use NICL API endpoint "
            "'https://api.prediction.neuralk-ai.com/api/v1/inference' instead."
        )
    return exc


def _resolve_nicl_token(token_env: str) -> str:
    token = os.getenv(token_env)
    if token:
        return token
    # Backward-compatible fallback for existing local setups.
    if token_env != "NICL_API_TOKEN":
        fallback = os.getenv("NICL_API_TOKEN")
        if fallback:
            return fallback
    raise AdapterSkipError(
        f"Missing NICL token in env var {token_env} (or NICL_API_TOKEN fallback)."
    )


def _compute_train_bin_centers(
    y_train: np.ndarray,
    y_train_cls: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    centers = np.zeros((num_classes,), dtype=np.float64)
    global_mean = float(np.mean(y_train)) if y_train.size else 0.0
    for cls in range(num_classes):
        mask = y_train_cls == cls
        if np.any(mask):
            centers[cls] = float(np.mean(y_train[mask]))
        else:
            centers[cls] = global_mean
    return centers


def _classification_outputs_to_regression(
    *,
    pred_cls: np.ndarray,
    proba: np.ndarray,
    centers: np.ndarray,
) -> np.ndarray:
    pred_cls_arr = np.asarray(pred_cls, dtype=np.int64).reshape(-1)
    proba_arr = np.asarray(proba, dtype=np.float64)

    if proba_arr.ndim == 2 and proba_arr.shape[1] == centers.shape[0]:
        pred = proba_arr @ centers
    else:
        safe_cls = np.clip(pred_cls_arr, 0, centers.shape[0] - 1)
        pred = centers[safe_cls]
    return np.asarray(pred, dtype=np.float64).reshape(-1)


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
    enabled = set(cfg.models.enabled_regression_models)
    adapters: dict[str, Any] = {}

    if "nanotabpfn_standard" in enabled:
        adapters["nanotabpfn_standard"] = NanoTabPFNForecastAdapter(
            name="nanotabpfn_standard",
            model_path=cfg.models.model_standard_ckpt,
            dist_path=cfg.models.model_standard_dist,
            device=device,
        )

    if "tabicl_regressor" in enabled:
        adapters["tabicl_regressor"] = TabICLForecastAdapter(
            checkpoint_version=cfg.models.tabicl_checkpoint_version,
            model_path=cfg.models.tabicl_model_path,
            device=device,
        )

    if "nanotabpfn_dynscm" in enabled:
        if (
            cfg.models.model_dynscm_ckpt is None
            and cfg.models.model_dynscm_dist is None
        ):
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

    if "nicl_regression" in enabled:
        if cfg.models.nicl_regression_mode == "off":
            adapters["nicl_regression"] = UnavailableRegressionAdapter(
                "nicl_regression",
                "Enable models.nicl_regression_mode to use NICL regression.",
            )
        elif cfg.models.nicl_regression_endpoint is None:
            adapters["nicl_regression"] = UnavailableRegressionAdapter(
                "nicl_regression",
                "nicl_regression_endpoint is not configured.",
            )
        else:
            adapters["nicl_regression"] = NICLRegressionAdapter(
                api_url=cfg.models.nicl_regression_endpoint,
                timeout_seconds=cfg.models.nicl_timeout_seconds,
                max_retries=cfg.models.nicl_max_retries,
                mode=cfg.models.nicl_regression_mode,
                token_env=cfg.models.nicl_api_key_env,
                model_name=cfg.models.nicl_model,
                proxy_num_classes=cfg.proxy.num_classes,
                min_samples_per_class=cfg.proxy.min_samples_per_class,
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
            token_env=cfg.models.nicl_api_key_env,
        ),
    }
