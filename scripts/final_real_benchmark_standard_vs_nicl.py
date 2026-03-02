#!/usr/bin/env python3
"""Run real-data benchmark (standard vs NICL) and print per-dataset metrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.prepare_forecast_datasets import prepare_dataset
from tfmplayground.benchmarks.forecasting.adapters import (
    NanoTabPFNForecastAdapter,
    NICLRegressionAdapter,
)
from tfmplayground.benchmarks.forecasting.config import ForecastBenchmarkConfig
from tfmplayground.benchmarks.forecasting.datasets import load_suite
from tfmplayground.benchmarks.forecasting.runner import evaluate_regression

NICL_API_ENDPOINT = "https://api.prediction.neuralk-ai.com/api/v1/inference"
TARGET_DATASETS = ("exchange_rate", "ettm1", "m4_weekly", "tourism_monthly")


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


def _ensure_token_available(repo_root: Path, env_name: str) -> str:
    token = os.getenv(env_name)
    source = env_name
    if not token:
        token = os.getenv("NICL_API_TOKEN")
        source = "NICL_API_TOKEN"
    if not token:
        dotenv = _load_dotenv_values(repo_root / ".env")
        token = dotenv.get(env_name) or dotenv.get("NICL_API_TOKEN")
        source = ".env"
        if token:
            os.environ[env_name] = token
    if not token:
        raise RuntimeError(
            f"Missing NICL API token in {env_name}, NICL_API_TOKEN, or .env."
        )
    return source


def _build_cfg(
    *,
    repo_root: Path,
    config_path: Path,
    output_dir: Path,
    cache_dir: Path,
    max_series_per_dataset: int | None,
) -> ForecastBenchmarkConfig:
    cfg = ForecastBenchmarkConfig.from_json(config_path)
    payload = cfg.to_dict()
    payload["mode"] = "regression"
    payload["output_dir"] = output_dir
    payload["datasets"]["dataset_names"] = TARGET_DATASETS
    payload["datasets"]["cache_dir"] = cache_dir
    if max_series_per_dataset is not None:
        payload["datasets"]["max_series_per_dataset"] = int(max_series_per_dataset)
    payload["models"]["enabled_regression_models"] = (
        "nanotabpfn_standard",
        "nicl_regression",
    )
    payload["models"]["nicl_regression_mode"] = "quantized_proxy"
    payload["models"]["nicl_regression_endpoint"] = NICL_API_ENDPOINT
    payload["models"]["nicl_api_url"] = NICL_API_ENDPOINT
    return ForecastBenchmarkConfig.from_dict(payload)


def _prepare_missing_caches(cfg: ForecastBenchmarkConfig, *, timeout: float) -> None:
    # exchange_rate + ettm1 are handled by load_suite auto-download.
    cache_dir = Path(cfg.datasets.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name in ("m4_weekly", "tourism_monthly"):
        prepare_dataset(
            dataset_name,
            cache_dir=cache_dir,
            timeout=timeout,
            force=False,
        )


def _build_adapters(cfg: ForecastBenchmarkConfig) -> dict[str, Any]:
    return {
        "nanotabpfn_standard": NanoTabPFNForecastAdapter(
            name="nanotabpfn_standard",
            model_path=cfg.models.model_standard_ckpt,
            dist_path=cfg.models.model_standard_dist,
            device="cpu",
        ),
        "nicl_regression": NICLRegressionAdapter(
            api_url=cfg.models.nicl_regression_endpoint or NICL_API_ENDPOINT,
            timeout_seconds=cfg.models.nicl_timeout_seconds,
            max_retries=cfg.models.nicl_max_retries,
            mode=cfg.models.nicl_regression_mode,
            token_env=cfg.models.nicl_api_key_env,
            model_name=cfg.models.nicl_model,
            proxy_num_classes=cfg.proxy.num_classes,
            min_samples_per_class=cfg.proxy.min_samples_per_class,
        ),
    }


def _compute_per_dataset_perf(rows: pd.DataFrame) -> pd.DataFrame:
    ok = rows.loc[rows["status"] == "ok"].copy()
    if ok.empty:
        return pd.DataFrame(
            columns=["dataset", "model", "rmse", "smape", "mase", "n_rows"]
        )
    perf = (
        ok.groupby(["dataset", "model"], as_index=False)
        .agg(
            rmse=("rmse", "mean"),
            smape=("smape", "mean"),
            mase=("mase", "mean"),
            n_rows=("rmse", "count"),
        )
        .sort_values(["dataset", "model"])
        .reset_index(drop=True)
    )
    return perf


def _status_summary(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["dataset", "model", "status"], as_index=False)
        .size()
        .sort_values(["dataset", "model", "status"])
        .reset_index(drop=True)
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run two-model real-data benchmark and show per-dataset performance."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/forecast_bench_final_3model.json"),
        help="Base benchmark config JSON.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("workdir/forecast_results_standard_vs_nicl_py"),
        help="Output directory for rows + summaries.",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path("workdir/forecast_data"),
        help="Dataset cache directory.",
    )
    parser.add_argument(
        "--max_series_per_dataset",
        type=int,
        default=8,
        help="Cap series per dataset for runtime control (default: 8).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Download timeout (seconds) for missing local caches.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else repo_root / args.output_dir
    )
    cache_dir = (
        args.cache_dir if args.cache_dir.is_absolute() else repo_root / args.cache_dir
    )

    cfg = _build_cfg(
        repo_root=repo_root,
        config_path=config_path,
        output_dir=output_dir,
        cache_dir=cache_dir,
        max_series_per_dataset=args.max_series_per_dataset,
    )
    token_source = _ensure_token_available(repo_root, cfg.models.nicl_api_key_env)

    print("Config path:", config_path)
    print("Output dir:", cfg.output_dir)
    print("Cache dir:", cfg.datasets.cache_dir)
    print("Datasets:", list(cfg.datasets.dataset_names))
    print("Max series per dataset:", cfg.datasets.max_series_per_dataset)
    print("NICL endpoint:", cfg.models.nicl_regression_endpoint)
    print("NICL token source:", token_source)

    _prepare_missing_caches(cfg, timeout=args.timeout)

    suite = load_suite(cfg)
    skipped_bundles = {
        name: bundle.skip_reason for name, bundle in suite.items() if bundle.skipped
    }
    if skipped_bundles:
        details = "; ".join(f"{k}: {v}" for k, v in skipped_bundles.items())
        raise RuntimeError(f"Dataset loading failed: {details}")

    rows = evaluate_regression(
        cfg, suite=suite, adapters=_build_adapters(cfg), device="cpu"
    )
    status = _status_summary(rows)
    perf = _compute_per_dataset_perf(rows)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "regression_rows_standard_vs_nicl.csv"
    perf_path = output_dir / "per_dataset_perf_standard_vs_nicl.csv"
    status_path = output_dir / "status_counts_standard_vs_nicl.csv"
    meta_path = output_dir / "run_metadata_standard_vs_nicl.json"

    rows.to_csv(rows_path, index=False)
    perf.to_csv(perf_path, index=False)
    status.to_csv(status_path, index=False)

    metadata = {
        "datasets": list(cfg.datasets.dataset_names),
        "models": ["nanotabpfn_standard", "nicl_regression"],
        "cache_dir": str(cfg.datasets.cache_dir),
        "output_dir": str(cfg.output_dir),
        "max_series_per_dataset": int(cfg.datasets.max_series_per_dataset),
        "nicl_endpoint": cfg.models.nicl_regression_endpoint,
        "token_source": token_source,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nStatus counts:")
    print(status.to_string(index=False))
    print("\nPer-dataset performance:")
    print(perf.to_string(index=False))

    bad = rows.loc[rows["status"] != "ok"]
    if not bad.empty:
        print("\nTop skip reasons:")
        print(
            bad[["dataset", "model", "series_id", "skip_reason"]]
            .head(30)
            .to_string(index=False)
        )
        raise RuntimeError("Some rows were skipped; see status/skip reasons above.")

    print("\nArtifacts:")
    print(rows_path)
    print(perf_path)
    print(status_path)
    print(meta_path)


if __name__ == "__main__":
    main()
