#!/usr/bin/env python3
"""Preflight validation for the final forecasting benchmark run."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from tfmplayground.benchmarks.forecasting.adapters import NanoTabPFNForecastAdapter
from tfmplayground.benchmarks.forecasting.config import ForecastBenchmarkConfig
from tfmplayground.benchmarks.forecasting.datasets import load_suite
from tfmplayground.interface import NanoTabPFNRegressor

FINAL_DATASETS = (
    "exchange_rate",
    "ettm1",
    "m4_weekly",
    "tourism_monthly",
)
FINAL_MODELS = (
    "nanotabpfn_standard",
    "nanotabpfn_dynscm",
    "nicl_regression",
)


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def _load_dotenv_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _run_check(name: str, fn: Callable[[], str]) -> CheckResult:
    try:
        return CheckResult(name=name, ok=True, detail=fn())
    except Exception as exc:
        return CheckResult(name=name, ok=False, detail=str(exc))


def _check_config(cfg: ForecastBenchmarkConfig) -> str:
    if tuple(cfg.datasets.dataset_names) != FINAL_DATASETS:
        got_datasets = tuple(cfg.datasets.dataset_names)
        raise ValueError(f"dataset_names must be {FINAL_DATASETS}, got {got_datasets}")
    if tuple(cfg.models.enabled_regression_models) != FINAL_MODELS:
        raise ValueError(
            "enabled_regression_models must be "
            f"{FINAL_MODELS}, got {tuple(cfg.models.enabled_regression_models)}"
        )
    if cfg.models.nicl_regression_mode != "quantized_proxy":
        raise ValueError(
            "nicl_regression_mode must be 'quantized_proxy' for final run."
        )
    if cfg.protocol.context_rows < 2 or cfg.protocol.test_rows < 1:
        raise ValueError("context_rows/test_rows are not feasible.")
    if min(cfg.protocol.horizons) <= 0:
        raise ValueError("horizons must be positive.")
    return "config matches final benchmark contract"


def _check_datasets(cfg: ForecastBenchmarkConfig) -> str:
    suite = load_suite(cfg)
    failed = {
        name: bundle.skip_reason for name, bundle in suite.items() if bundle.skipped
    }
    if failed:
        detail = "; ".join(f"{name}: {reason}" for name, reason in failed.items())
        raise RuntimeError(detail)
    return f"loaded {len(suite)} datasets"


def _check_standard_model() -> str:
    model = NanoTabPFNRegressor(model=None, dist=None, device="cpu")
    return f"loaded {type(model).__name__} via default pretrained checkpoint"


def _check_dynscm_model(cfg: ForecastBenchmarkConfig) -> str:
    if cfg.models.model_dynscm_ckpt is None:
        raise ValueError("models.model_dynscm_ckpt must be set.")
    ckpt_path = Path(cfg.models.model_dynscm_ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt_path}")
    if cfg.models.model_dynscm_dist is not None:
        dist_path = Path(cfg.models.model_dynscm_dist)
        if not dist_path.exists():
            raise FileNotFoundError(f"missing bucket file: {dist_path}")

    adapter = NanoTabPFNForecastAdapter(
        name="nanotabpfn_dynscm",
        model_path=cfg.models.model_dynscm_ckpt,
        dist_path=cfg.models.model_dynscm_dist,
        device="cpu",
    )
    x_train = np.zeros((4, 3), dtype=np.float64)
    y_train = np.linspace(0.0, 1.0, 4, dtype=np.float64)
    x_test = np.zeros((2, 3), dtype=np.float64)
    pred = adapter.fit_predict(x_train, y_train, x_test)
    if pred.shape != (2,):
        raise ValueError(f"unexpected prediction shape {pred.shape}")
    return f"loaded checkpoint {ckpt_path.name}"


def _check_nicl(cfg: ForecastBenchmarkConfig, dotenv_values: dict[str, str]) -> str:
    endpoint = cfg.models.nicl_regression_endpoint
    if not endpoint:
        raise ValueError("models.nicl_regression_endpoint must be set.")
    _validate_nicl_endpoint(endpoint)
    env_name = cfg.models.nicl_api_key_env
    token = os.getenv(env_name) or os.getenv("NICL_API_TOKEN")
    source = env_name if os.getenv(env_name) else "NICL_API_TOKEN"
    if token is None:
        token = dotenv_values.get(env_name) or dotenv_values.get("NICL_API_TOKEN")
        source = ".env"
    if not token:
        raise ValueError(
            f"missing NICL token in {env_name}, NICL_API_TOKEN, or local .env"
        )
    budget = cfg.models.nicl_max_rows_budget
    budget_text = f", budget={budget}" if budget is not None else ""
    return f"endpoint configured ({endpoint}), token found via {source}{budget_text}"


def _validate_nicl_endpoint(endpoint: str) -> None:
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "models.nicl_regression_endpoint must be an absolute http(s) URL."
        )
    normalized = endpoint.rstrip("/")
    if normalized == "https://prediction.neuralk-ai.com/predict":
        raise ValueError(
            "models.nicl_regression_endpoint points to dashboard URL "
            "https://prediction.neuralk-ai.com/predict. "
            "Use https://api.prediction.neuralk-ai.com/api/v1/inference instead."
        )
    if parsed.path.rstrip("/") != "/api/v1/inference":
        raise ValueError(
            "models.nicl_regression_endpoint must end with /api/v1/inference "
            f"(got {parsed.path or '/'})."
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check whether the final forecasting benchmark is ready to run."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/forecast_bench_final_3model.json"),
        help="Final benchmark config JSON.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    cfg = ForecastBenchmarkConfig.from_json(args.config)
    dotenv_values = _load_dotenv_values(Path(".env"))

    results = [
        _run_check("config", lambda: _check_config(cfg)),
        _run_check("datasets", lambda: _check_datasets(cfg)),
        _run_check("standard_model", _check_standard_model),
        _run_check("dynscm_model", lambda: _check_dynscm_model(cfg)),
        _run_check("nicl", lambda: _check_nicl(cfg, dotenv_values)),
    ]

    failed = [result for result in results if not result.ok]
    print("PASS" if not failed else "FAIL")
    for result in results:
        prefix = "PASS" if result.ok else "FAIL"
        print(f"[{prefix}] {result.name}: {result.detail}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
