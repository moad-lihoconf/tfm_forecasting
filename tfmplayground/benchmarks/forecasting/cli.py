"""Command-line interface for forecasting benchmark runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import ForecastBenchmarkConfig
from .runner import run_benchmark


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run forecasting benchmark validation for DynSCM research claims."
    )
    parser.add_argument(
        "--mode",
        choices=["regression", "proxy", "both"],
        default="both",
        help="Benchmark track(s) to execute.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to ForecastBenchmarkConfig JSON.",
    )
    parser.add_argument(
        "--model_standard_ckpt",
        type=str,
        default=None,
        help="Checkpoint path for standard nanoTabPFN regressor baseline.",
    )
    parser.add_argument(
        "--model_standard_dist",
        type=str,
        default=None,
        help="Bucket edges path for standard nanoTabPFN regressor baseline.",
    )
    parser.add_argument(
        "--model_dynscm_ckpt",
        type=str,
        default=None,
        help="Checkpoint path for DynSCM-trained nanoTabPFN regressor.",
    )
    parser.add_argument(
        "--model_dynscm_dist",
        type=str,
        default=None,
        help="Bucket edges path for DynSCM-trained nanoTabPFN regressor.",
    )
    parser.add_argument(
        "--tabicl_checkpoint_version",
        type=str,
        default=None,
        help="TabICL checkpoint version identifier.",
    )
    parser.add_argument(
        "--nicl_regression_mode",
        choices=["off", "native", "quantized_proxy"],
        default=None,
        help="Enable NICL regression adapter mode.",
    )
    parser.add_argument(
        "--nicl_regression_endpoint",
        type=str,
        default=None,
        help="NICL regression endpoint URL.",
    )
    parser.add_argument(
        "--nicl_max_rows_budget",
        type=int,
        default=None,
        help="Optional total rows budget for NICL regression calls.",
    )
    parser.add_argument(
        "--nicl_fail_on_unavailable",
        action="store_true",
        help="Fail benchmark if NICL regression is enabled but unavailable.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory where benchmark artifacts are written.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for local adapters.",
    )
    return parser


def _load_config(args: argparse.Namespace) -> ForecastBenchmarkConfig:
    cfg = (
        ForecastBenchmarkConfig.from_json(args.config)
        if args.config is not None
        else ForecastBenchmarkConfig()
    )

    payload = cfg.to_dict()
    payload["mode"] = args.mode

    if args.output_dir is not None:
        payload["output_dir"] = Path(args.output_dir)

    models_payload = dict(payload["models"])
    for arg_name, model_key in (
        ("model_standard_ckpt", "model_standard_ckpt"),
        ("model_standard_dist", "model_standard_dist"),
        ("model_dynscm_ckpt", "model_dynscm_ckpt"),
        ("model_dynscm_dist", "model_dynscm_dist"),
        ("tabicl_checkpoint_version", "tabicl_checkpoint_version"),
        ("nicl_regression_mode", "nicl_regression_mode"),
        ("nicl_regression_endpoint", "nicl_regression_endpoint"),
        ("nicl_max_rows_budget", "nicl_max_rows_budget"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            models_payload[model_key] = value
    if args.nicl_fail_on_unavailable:
        models_payload["nicl_fail_on_unavailable"] = True
    payload["models"] = models_payload

    return ForecastBenchmarkConfig.from_dict(payload)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cfg = _load_config(args)
    artifacts = run_benchmark(cfg, device=args.device)

    print(f"Report: {artifacts.report_path}")
    if artifacts.regression_rows_path is not None:
        print(f"Regression rows: {artifacts.regression_rows_path}")
    if artifacts.regression_summary_path is not None:
        print(f"Regression summary: {artifacts.regression_summary_path}")
    if artifacts.proxy_rows_path is not None:
        print(f"Proxy rows: {artifacts.proxy_rows_path}")
    if artifacts.proxy_summary_path is not None:
        print(f"Proxy summary: {artifacts.proxy_summary_path}")


__all__ = ["main"]
