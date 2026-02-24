"""Benchmark runner for forecasting research validation."""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_CANDIDATE_MODEL = "nanotabpfn_dynscm"
_BASELINE_MODELS = ("nanotabpfn_standard", "tabicl_regressor")
_PRIMARY_BASELINE = "nanotabpfn_standard"
_NICL_REGRESSION_MODEL = "nicl_regression"

from .adapters import (
    AdapterSkipError,
    build_forecast_table_from_series,
    default_proxy_adapters,
    default_regression_adapters,
)
from .config import ForecastBenchmarkConfig
from .datasets import DatasetBundle, load_suite
from .metrics import (
    bootstrap_ci,
    compute_proxy_metrics,
    compute_regression_metrics,
    relative_improvement,
)
from .protocol import generate_rolling_origin_indices
from .proxy_classification import fit_quantile_binner, transform_to_classes
from .report import build_markdown_report, write_json


@dataclass(frozen=True, slots=True)
class BenchmarkArtifacts:
    """Paths and in-memory summaries for benchmark outputs."""

    regression_rows_path: Path | None
    regression_summary_path: Path | None
    proxy_rows_path: Path | None
    proxy_summary_path: Path | None
    report_path: Path
    regression_rows: pd.DataFrame | None
    regression_summary: dict[str, Any] | None
    proxy_rows: pd.DataFrame | None
    proxy_summary: dict[str, Any] | None


def evaluate_regression(
    cfg: ForecastBenchmarkConfig,
    *,
    suite: dict[str, DatasetBundle] | None = None,
    adapters: dict[str, Any] | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    """Evaluate regression baselines on loaded suite using shared protocol."""
    loaded_suite = load_suite(cfg) if suite is None else suite
    model_adapters = (
        default_regression_adapters(cfg, device=device)
        if adapters is None
        else adapters
    )
    _validate_nicl_regression_availability(cfg=cfg, adapters=model_adapters)

    rows: list[dict[str, Any]] = []
    required_lag = cfg.protocol.required_lag
    nicl_budget_rows = cfg.models.nicl_max_rows_budget
    nicl_rows_used = 0

    for dataset_name, bundle in loaded_suite.items():
        if bundle.skipped:
            for model_name in model_adapters:
                rows.append(
                    _regression_skip_row(
                        dataset=dataset_name,
                        model=model_name,
                        reason=bundle.skip_reason or "dataset skipped",
                    )
                )
            continue

        for series_id, raw_series in enumerate(bundle.series):
            series = np.asarray(raw_series, dtype=np.float64)
            if series.ndim != 1:
                rows.append(
                    _regression_skip_row(
                        dataset=dataset_name,
                        model="all",
                        reason=f"Series {series_id} has invalid shape {series.shape}",
                        series_id=series_id,
                    )
                )
                continue

            try:
                indices = generate_rolling_origin_indices(
                    series_length=series.size,
                    horizons=cfg.protocol.horizons,
                    n_train=cfg.protocol.context_rows,
                    n_test=cfg.protocol.test_rows,
                    required_lag=required_lag,
                    seed=cfg.seed + series_id,
                )
                table = build_forecast_table_from_series(
                    cfg,
                    series,
                    t_idx=indices.t_idx,
                    h_idx=indices.h_idx,
                    split_index=indices.split_index,
                    seed=cfg.seed + series_id,
                )
            except Exception as exc:
                for model_name in model_adapters:
                    rows.append(
                        _regression_skip_row(
                            dataset=dataset_name,
                            model=model_name,
                            reason=f"Failed preprocessing series {series_id}: {exc}",
                            series_id=series_id,
                        )
                    )
                continue

            split = int(table.split_index)
            x_train = table.x[:split]
            y_train = table.y[:split]
            x_test = table.x[split:]
            y_test = table.y[split:]
            h_test = table.h_idx[split:]

            max_train_origin = int(np.max(table.t_idx[:split]))
            insample = _fill_nan_1d(series[: max_train_origin + 1])

            for model_name, model in model_adapters.items():
                if (
                    model_name == _NICL_REGRESSION_MODEL
                    and nicl_budget_rows is not None
                ):
                    rows_in_call = int(x_train.shape[0] + x_test.shape[0])
                    if nicl_rows_used + rows_in_call > nicl_budget_rows:
                        rows.append(
                            _regression_skip_row(
                                dataset=dataset_name,
                                model=model_name,
                                reason=(
                                    "NICL regression rows budget exceeded: "
                                    f"used={nicl_rows_used}, need={rows_in_call}, "
                                    f"budget={nicl_budget_rows}"
                                ),
                                series_id=series_id,
                            )
                        )
                        continue
                try:
                    pred = np.asarray(
                        model.fit_predict(x_train, y_train, x_test), dtype=np.float64
                    )
                    if pred.shape != y_test.shape:
                        raise ValueError(
                            f"Prediction shape {pred.shape} does not match target shape {y_test.shape}."
                        )
                except AdapterSkipError as exc:
                    rows.append(
                        _regression_skip_row(
                            dataset=dataset_name,
                            model=model_name,
                            reason=str(exc),
                            series_id=series_id,
                        )
                    )
                    continue

                if (
                    model_name == _NICL_REGRESSION_MODEL
                    and nicl_budget_rows is not None
                ):
                    nicl_rows_used += int(x_train.shape[0] + x_test.shape[0])

                for horizon in sorted(set(int(v) for v in h_test.tolist())):
                    mask = h_test == horizon
                    if not np.any(mask):
                        continue
                    metric_values = compute_regression_metrics(
                        y_test[mask],
                        pred[mask],
                        insample=insample,
                        seasonality=bundle.seasonality,
                    )
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "frequency": bundle.frequency,
                            "seasonality": int(bundle.seasonality),
                            "series_id": int(series_id),
                            "model": model_name,
                            "horizon": int(horizon),
                            "n_points": int(np.sum(mask)),
                            "rmse": float(metric_values["rmse"]),
                            "smape": float(metric_values["smape"]),
                            "mase": float(metric_values["mase"]),
                            "status": "ok",
                            "skip_reason": "",
                        }
                    )

    return pd.DataFrame(rows)


def evaluate_proxy(
    cfg: ForecastBenchmarkConfig,
    *,
    suite: dict[str, DatasetBundle] | None = None,
    adapters: dict[str, Any] | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    """Evaluate proxy classification baselines on shared forecast tables."""
    loaded_suite = load_suite(cfg) if suite is None else suite
    model_adapters = (
        default_proxy_adapters(cfg, device=device) if adapters is None else adapters
    )

    rows: list[dict[str, Any]] = []
    required_lag = cfg.protocol.required_lag

    for dataset_name, bundle in loaded_suite.items():
        if bundle.skipped:
            for model_name in model_adapters:
                rows.append(
                    _proxy_skip_row(
                        dataset=dataset_name,
                        model=model_name,
                        reason=bundle.skip_reason or "dataset skipped",
                    )
                )
            continue

        for series_id, raw_series in enumerate(bundle.series):
            series = np.asarray(raw_series, dtype=np.float64)
            try:
                indices = generate_rolling_origin_indices(
                    series_length=series.size,
                    horizons=cfg.protocol.horizons,
                    n_train=cfg.protocol.context_rows,
                    n_test=cfg.protocol.test_rows,
                    required_lag=required_lag,
                    seed=cfg.seed + 100_000 + series_id,
                )
                table = build_forecast_table_from_series(
                    cfg,
                    series,
                    t_idx=indices.t_idx,
                    h_idx=indices.h_idx,
                    split_index=indices.split_index,
                    seed=cfg.seed + 100_000 + series_id,
                )
            except Exception as exc:
                for model_name in model_adapters:
                    rows.append(
                        _proxy_skip_row(
                            dataset=dataset_name,
                            model=model_name,
                            reason=f"Failed preprocessing series {series_id}: {exc}",
                            series_id=series_id,
                        )
                    )
                continue

            split = int(table.split_index)
            x_train = table.x[:split]
            y_train_cont = table.y[:split]
            x_test = table.x[split:]
            y_test_cont = table.y[split:]
            h_test = table.h_idx[split:]

            try:
                bin_edges = fit_quantile_binner(
                    y_train_cont,
                    num_classes=cfg.proxy.num_classes,
                    min_samples_per_class=cfg.proxy.min_samples_per_class,
                )
                y_train = transform_to_classes(y_train_cont, bin_edges)
                y_test = transform_to_classes(y_test_cont, bin_edges)
            except Exception as exc:
                for model_name in model_adapters:
                    rows.append(
                        _proxy_skip_row(
                            dataset=dataset_name,
                            model=model_name,
                            reason=f"Failed proxy binning series {series_id}: {exc}",
                            series_id=series_id,
                        )
                    )
                continue

            num_classes = max(2, int(bin_edges.size - 1))
            for model_name, model in model_adapters.items():
                try:
                    if _accepts_num_classes(model):
                        y_pred, y_proba = model.fit_predict_proba(
                            x_train,
                            y_train,
                            x_test,
                            num_classes=num_classes,
                        )
                    else:
                        y_pred, y_proba = model.fit_predict_proba(
                            x_train, y_train, x_test
                        )
                except AdapterSkipError as exc:
                    rows.append(
                        _proxy_skip_row(
                            dataset=dataset_name,
                            model=model_name,
                            reason=str(exc),
                            series_id=series_id,
                        )
                    )
                    continue

                y_pred = np.asarray(y_pred, dtype=np.int64)
                y_proba_arr = np.asarray(y_proba, dtype=np.float64)
                if y_pred.shape != y_test.shape:
                    rows.append(
                        _proxy_skip_row(
                            dataset=dataset_name,
                            model=model_name,
                            reason=(
                                f"Prediction shape mismatch for series {series_id}: "
                                f"pred={y_pred.shape}, target={y_test.shape}"
                            ),
                            series_id=series_id,
                        )
                    )
                    continue

                for horizon in sorted(set(int(v) for v in h_test.tolist())):
                    mask = h_test == horizon
                    if not np.any(mask):
                        continue
                    metric_values = compute_proxy_metrics(
                        y_true=y_test[mask],
                        y_pred=y_pred[mask],
                        y_proba=y_proba_arr[mask]
                        if y_proba_arr.ndim > 1
                        else y_proba_arr[mask],
                    )
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "frequency": bundle.frequency,
                            "seasonality": int(bundle.seasonality),
                            "series_id": int(series_id),
                            "model": model_name,
                            "horizon": int(horizon),
                            "n_points": int(np.sum(mask)),
                            "balanced_accuracy": float(
                                metric_values["balanced_accuracy"]
                            ),
                            "macro_auroc": float(metric_values["macro_auroc"]),
                            "status": "ok",
                            "skip_reason": "",
                        }
                    )

    return pd.DataFrame(rows)


def run_benchmark(
    cfg: ForecastBenchmarkConfig,
    *,
    device: str = "cpu",
) -> BenchmarkArtifacts:
    """Run configured benchmark tracks and write all artifacts."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    regression_rows: pd.DataFrame | None = None
    regression_summary: dict[str, Any] | None = None
    regression_rows_path: Path | None = None
    regression_summary_path: Path | None = None

    proxy_rows: pd.DataFrame | None = None
    proxy_summary: dict[str, Any] | None = None
    proxy_rows_path: Path | None = None
    proxy_summary_path: Path | None = None

    suite = load_suite(cfg)
    nicl_capabilities = _build_nicl_capabilities(cfg)
    write_json(output_dir / "nicl_capabilities.json", nicl_capabilities)

    if cfg.mode in {"regression", "both"}:
        regression_rows = evaluate_regression(cfg, suite=suite, device=device)
        regression_summary = summarize_regression(regression_rows, cfg)
        regression_summary["nicl_capabilities"] = nicl_capabilities

        regression_rows_path = output_dir / "regression_rows.csv"
        regression_summary_path = output_dir / "regression_summary.json"
        regression_rows.to_csv(regression_rows_path, index=False)
        write_json(regression_summary_path, regression_summary)

    if cfg.mode in {"proxy", "both"}:
        proxy_rows = evaluate_proxy(cfg, suite=suite, device=device)
        proxy_summary = summarize_proxy(proxy_rows)

        proxy_rows_path = output_dir / "proxy_rows.csv"
        proxy_summary_path = output_dir / "proxy_summary.json"
        proxy_rows.to_csv(proxy_rows_path, index=False)
        write_json(proxy_summary_path, proxy_summary)

    report_text = build_markdown_report(
        regression_summary=regression_summary,
        proxy_summary=proxy_summary,
    )
    report_path = output_dir / "report.md"
    report_path.write_text(report_text, encoding="utf-8")

    return BenchmarkArtifacts(
        regression_rows_path=regression_rows_path,
        regression_summary_path=regression_summary_path,
        proxy_rows_path=proxy_rows_path,
        proxy_summary_path=proxy_summary_path,
        report_path=report_path,
        regression_rows=regression_rows,
        regression_summary=regression_summary,
        proxy_rows=proxy_rows,
        proxy_summary=proxy_summary,
    )


def summarize_regression(
    rows: pd.DataFrame,
    cfg: ForecastBenchmarkConfig,
) -> dict[str, Any]:
    """Compute win-rate + bootstrap CI summaries and claim status."""
    if rows.empty:
        return {
            "comparisons": [],
            "claim": {
                "primary_claim_pass": False,
                "required_metric_passes": int(cfg.stats.min_metrics_to_pass),
                "achieved_metric_passes": 0,
            },
        }

    ok_rows = rows.loc[rows["status"] == "ok"].copy()
    if ok_rows.empty:
        return {
            "comparisons": [],
            "claim": {
                "primary_claim_pass": False,
                "required_metric_passes": int(cfg.stats.min_metrics_to_pass),
                "achieved_metric_passes": 0,
            },
        }

    candidate_model = _CANDIDATE_MODEL
    baselines = _BASELINE_MODELS
    key_cols = ["dataset", "series_id", "horizon"]

    comparisons: list[dict[str, Any]] = []
    achieved_passes_vs_standard = 0

    for baseline_model in baselines:
        base_rows = ok_rows.loc[ok_rows["model"] == baseline_model]
        cand_rows = ok_rows.loc[ok_rows["model"] == candidate_model]
        if base_rows.empty or cand_rows.empty:
            for metric_name in cfg.stats.claim_metrics:
                comparisons.append(
                    {
                        "baseline": baseline_model,
                        "candidate": candidate_model,
                        "metric": metric_name,
                        "mean_improvement": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "win_rate": np.nan,
                        "n_pairs": 0,
                        "pass": False,
                    }
                )
            continue

        merged = base_rows.merge(
            cand_rows,
            on=key_cols,
            suffixes=("_base", "_cand"),
        )
        if merged.empty:
            for metric_name in cfg.stats.claim_metrics:
                comparisons.append(
                    {
                        "baseline": baseline_model,
                        "candidate": candidate_model,
                        "metric": metric_name,
                        "mean_improvement": np.nan,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "win_rate": np.nan,
                        "n_pairs": 0,
                        "pass": False,
                    }
                )
            continue

        for metric_name in cfg.stats.claim_metrics:
            base_metric = merged[f"{metric_name}_base"].to_numpy(dtype=np.float64)
            cand_metric = merged[f"{metric_name}_cand"].to_numpy(dtype=np.float64)
            improvement = relative_improvement(base_metric, cand_metric)

            mean, ci_low, ci_high = bootstrap_ci(
                improvement,
                n_bootstrap=cfg.stats.bootstrap_samples,
                confidence_level=cfg.stats.confidence_level,
                seed=cfg.seed,
            )
            win_rate = float(np.mean(improvement > 0.0)) if improvement.size else np.nan
            passed = bool(np.isfinite(ci_low) and ci_low > 0.0)

            if baseline_model == _PRIMARY_BASELINE and passed:
                achieved_passes_vs_standard += 1

            comparisons.append(
                {
                    "baseline": baseline_model,
                    "candidate": candidate_model,
                    "metric": metric_name,
                    "mean_improvement": float(mean),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "win_rate": float(win_rate),
                    "n_pairs": int(improvement.size),
                    "pass": passed,
                }
            )

    claim = {
        "primary_claim_pass": achieved_passes_vs_standard
        >= int(cfg.stats.min_metrics_to_pass),
        "required_metric_passes": int(cfg.stats.min_metrics_to_pass),
        "achieved_metric_passes": int(achieved_passes_vs_standard),
    }

    return {
        "comparisons": comparisons,
        "claim": claim,
        "nicl_regression": _summarize_nicl_regression(rows),
    }


def summarize_proxy(rows: pd.DataFrame) -> dict[str, Any]:
    """Compute aggregated proxy classification summary by model."""
    if rows.empty:
        return {"models": []}

    ok_rows = rows.loc[rows["status"] == "ok"].copy()
    if ok_rows.empty:
        return {"models": []}

    grouped = (
        ok_rows.groupby("model", as_index=False)
        .agg(
            balanced_accuracy=("balanced_accuracy", "mean"),
            macro_auroc=("macro_auroc", "mean"),
            n_rows=("balanced_accuracy", "count"),
        )
        .sort_values(by=["balanced_accuracy", "macro_auroc"], ascending=False)
    )

    return {"models": grouped.to_dict(orient="records")}


def _regression_skip_row(
    *,
    dataset: str,
    model: str,
    reason: str,
    series_id: int = -1,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "frequency": "unknown",
        "seasonality": np.nan,
        "series_id": int(series_id),
        "model": model,
        "horizon": np.nan,
        "n_points": 0,
        "rmse": np.nan,
        "smape": np.nan,
        "mase": np.nan,
        "status": "skipped",
        "skip_reason": reason,
    }


def _proxy_skip_row(
    *,
    dataset: str,
    model: str,
    reason: str,
    series_id: int = -1,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "frequency": "unknown",
        "seasonality": np.nan,
        "series_id": int(series_id),
        "model": model,
        "horizon": np.nan,
        "n_points": 0,
        "balanced_accuracy": np.nan,
        "macro_auroc": np.nan,
        "status": "skipped",
        "skip_reason": reason,
    }


def _accepts_num_classes(adapter: Any) -> bool:
    """Check if an adapter's fit_predict_proba accepts a num_classes kwarg."""
    method = getattr(adapter, "fit_predict_proba", None)
    if method is None:
        return False
    sig = inspect.signature(method)
    return "num_classes" in sig.parameters


def _summarize_nicl_regression(rows: pd.DataFrame) -> dict[str, Any]:
    nicl_rows = rows.loc[rows["model"] == _NICL_REGRESSION_MODEL]
    if nicl_rows.empty:
        return {
            "present": False,
            "ok_rows": 0,
            "skipped_rows": 0,
            "top_skip_reasons": [],
        }
    ok_rows = nicl_rows.loc[nicl_rows["status"] == "ok"]
    skipped = nicl_rows.loc[nicl_rows["status"] == "skipped"]
    top_reasons = (
        skipped["skip_reason"].value_counts().head(5).to_dict()
        if not skipped.empty
        else {}
    )
    return {
        "present": True,
        "ok_rows": int(len(ok_rows)),
        "skipped_rows": int(len(skipped)),
        "top_skip_reasons": [
            {"reason": str(reason), "count": int(count)}
            for reason, count in top_reasons.items()
        ],
    }


def _validate_nicl_regression_availability(
    *,
    cfg: ForecastBenchmarkConfig,
    adapters: dict[str, Any],
) -> None:
    if cfg.models.nicl_regression_mode == "off":
        return
    nicl_adapter = adapters.get(_NICL_REGRESSION_MODEL)
    unavailable = (
        nicl_adapter is None
        or nicl_adapter.__class__.__name__ == "UnavailableRegressionAdapter"
    )
    if unavailable and cfg.models.nicl_fail_on_unavailable:
        raise RuntimeError(
            "NICL regression is enabled but unavailable; "
            "set models.nicl_regression_endpoint and auth token, "
            "or disable strict mode (nicl_fail_on_unavailable=False)."
        )


def _build_nicl_capabilities(cfg: ForecastBenchmarkConfig) -> dict[str, Any]:
    token_primary = os.getenv(cfg.models.nicl_api_key_env)
    token_fallback = os.getenv("NICL_API_TOKEN")
    return {
        "regression_mode": cfg.models.nicl_regression_mode,
        "regression_endpoint": cfg.models.nicl_regression_endpoint,
        "api_key_env": cfg.models.nicl_api_key_env,
        "token_present_primary_env": bool(token_primary),
        "token_present_nicl_api_token_fallback": bool(token_fallback),
        "max_rows_budget": cfg.models.nicl_max_rows_budget,
        "strict_unavailable": cfg.models.nicl_fail_on_unavailable,
    }


def _fill_nan_1d(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("Expected a 1D array.")
    if arr.size == 0:
        return np.zeros((1,), dtype=np.float64)
    if np.isfinite(arr).all():
        return arr

    out = arr.copy()
    finite_mask = np.isfinite(out)
    if not finite_mask.any():
        return np.zeros_like(out)

    finite_idx = np.where(finite_mask)[0]
    finite_val = out[finite_mask]
    miss_idx = np.where(~finite_mask)[0]
    out[miss_idx] = np.interp(miss_idx, finite_idx, finite_val)
    return out
