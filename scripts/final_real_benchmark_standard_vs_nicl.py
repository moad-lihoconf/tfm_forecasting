#!/usr/bin/env python3
"""Unified reproducible benchmark runner.

Modes:
1) full: run expensive models (`nanotabpfn_standard` + `nicl_regression`)
2) append_stats: append cheap statistical baselines from cached data only

This keeps one shareable script for both workflows.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.prepare_forecast_datasets import prepare_dataset
from tfmplayground.benchmarks.forecasting.adapters import (
    NanoTabPFNForecastAdapter,
    NICLRegressionAdapter,
)
from tfmplayground.benchmarks.forecasting.config import ForecastBenchmarkConfig
from tfmplayground.benchmarks.forecasting.datasets import DatasetBundle, load_suite
from tfmplayground.benchmarks.forecasting.metrics import compute_regression_metrics
from tfmplayground.benchmarks.forecasting.protocol import (
    generate_rolling_origin_indices,
)
from tfmplayground.benchmarks.forecasting.runner import evaluate_regression

try:  # pragma: no cover - optional dependency
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:  # pragma: no cover
    ExponentialSmoothing = None
    ConvergenceWarning = Warning

NICL_API_ENDPOINT = "https://api.prediction.neuralk-ai.com/api/v1/inference"
TARGET_DATASETS = ("exchange_rate", "ettm1", "m4_weekly", "tourism_monthly")
BASE_PREFIX = "standard_vs_nicl"
NAIVE_LAST = "naive_last"
AUTO_ETS = "auto_ets"
STAT_MODEL_CHOICES = (NAIVE_LAST, AUTO_ETS)


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


def _resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _build_full_cfg(
    *,
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
        "nanotabpfn_dynscm",
        "nicl_regression",
    )
    payload["models"]["nicl_regression_mode"] = "quantized_proxy"
    payload["models"]["nicl_regression_endpoint"] = NICL_API_ENDPOINT
    payload["models"]["nicl_api_url"] = NICL_API_ENDPOINT
    return ForecastBenchmarkConfig.from_dict(payload)


def _build_stats_cfg(
    *,
    config_path: Path,
    cache_dir: Path,
    dataset_names: tuple[str, ...],
    max_series_per_dataset: int,
) -> ForecastBenchmarkConfig:
    cfg = ForecastBenchmarkConfig.from_json(config_path)
    payload = cfg.to_dict()
    payload["mode"] = "regression"
    payload["datasets"]["dataset_names"] = dataset_names
    payload["datasets"]["cache_dir"] = cache_dir
    payload["datasets"]["max_series_per_dataset"] = int(max_series_per_dataset)
    # Keep validator satisfied; these adapters are not used in append_stats mode.
    payload["models"]["enabled_regression_models"] = (
        "nanotabpfn_standard",
        "nanotabpfn_dynscm",
    )
    return ForecastBenchmarkConfig.from_dict(payload)


def _prepare_missing_caches(cfg: ForecastBenchmarkConfig, *, timeout: float) -> None:
    # exchange_rate + ettm1 are handled by load_suite auto-download.
    cache_dir = Path(cfg.datasets.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name in ("m4_weekly", "tourism_monthly"):
        if dataset_name in cfg.datasets.dataset_names:
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
    return (
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


def _status_summary(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["dataset", "model", "status"], as_index=False)
        .size()
        .sort_values(["dataset", "model", "status"])
        .reset_index(drop=True)
    )


def _rows_path(output_dir: Path, prefix: str) -> Path:
    return output_dir / f"regression_rows_{prefix}.csv"


def _perf_path(output_dir: Path, prefix: str) -> Path:
    return output_dir / f"per_dataset_perf_{prefix}.csv"


def _status_path(output_dir: Path, prefix: str) -> Path:
    return output_dir / f"status_counts_{prefix}.csv"


def _meta_path(output_dir: Path, prefix: str) -> Path:
    return output_dir / f"run_metadata_{prefix}.json"


def _normalize_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype(str).str.strip()
    numeric_cols = (
        "seasonality",
        "series_id",
        "horizon",
        "n_points",
        "rmse",
        "smape",
        "mase",
    )
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "skip_reason" in out.columns:
        out["skip_reason"] = out["skip_reason"].fillna("")
    if "status" in out.columns:
        out["status"] = out["status"].fillna("")
    return out


def _infer_series_ids_by_dataset(rows: pd.DataFrame) -> dict[str, set[int]]:
    if "dataset" not in rows.columns or "series_id" not in rows.columns:
        return {}
    valid = rows.loc[rows["series_id"].notna() & (rows["series_id"] >= 0)].copy()
    if valid.empty:
        return {}
    return {
        str(dataset): set(int(v) for v in group["series_id"].tolist())
        for dataset, group in valid.groupby("dataset", sort=False)
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


def _skip_row(
    *,
    dataset: str,
    frequency: str,
    seasonality: int,
    series_id: int,
    model: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "frequency": frequency,
        "seasonality": int(seasonality),
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


def _fit_auto_ets_and_forecast(
    *,
    train: np.ndarray,
    horizon: int,
    seasonality: int,
) -> np.ndarray:
    if ExponentialSmoothing is None:
        raise RuntimeError("statsmodels is not installed.")

    seasonal_periods = int(seasonality) if seasonality > 1 else 0
    if seasonal_periods < 2 or train.size < seasonal_periods * 2:
        seasonal_periods = 0

    candidates: list[tuple[str | None, str | None, bool]] = [
        (None, None, False),
        ("add", None, False),
        ("add", None, True),
    ]
    if seasonal_periods > 0:
        candidates.extend(
            [
                (None, "add", False),
                ("add", "add", False),
                ("add", "add", True),
            ]
        )

    best_result = None
    best_score = np.inf
    for trend, seasonal, damped in candidates:
        if trend is None and damped:
            continue
        try:
            model = ExponentialSmoothing(
                train,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods if seasonal is not None else None,
                damped_trend=damped if trend is not None else False,
                initialization_method="estimated",
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                fitted = model.fit(
                    optimized=True,
                    use_brute=False,
                    remove_bias=False,
                )
            score = float(getattr(fitted, "aic", np.inf))
            if not np.isfinite(score):
                score = float(getattr(fitted, "sse", np.inf))
            if score < best_score:
                best_score = score
                best_result = fitted
        except Exception:
            continue

    if best_result is None:
        return np.full((horizon,), float(train[-1]), dtype=np.float64)

    try:
        forecast = np.asarray(
            best_result.forecast(horizon),
            dtype=np.float64,
        ).reshape(-1)
    except Exception:
        return np.full((horizon,), float(train[-1]), dtype=np.float64)

    if forecast.size != horizon:
        return np.resize(forecast, (horizon,)).astype(np.float64, copy=False)
    return forecast


def _auto_ets_predict(
    *,
    series_filled: np.ndarray,
    t_test: np.ndarray,
    h_test: np.ndarray,
    seasonality: int,
) -> np.ndarray:
    predictions = np.zeros_like(h_test, dtype=np.float64)
    max_horizon_by_origin: dict[int, int] = {}
    row_ids_by_origin: dict[int, list[int]] = {}

    for row_idx, (origin, horizon) in enumerate(
        zip(t_test.tolist(), h_test.tolist(), strict=True)
    ):
        origin_int = int(origin)
        horizon_int = int(horizon)
        max_horizon_by_origin[origin_int] = max(
            max_horizon_by_origin.get(origin_int, 1),
            horizon_int,
        )
        row_ids_by_origin.setdefault(origin_int, []).append(row_idx)

    for origin, row_ids in row_ids_by_origin.items():
        train = series_filled[: origin + 1]
        if train.size < 4:
            forecast = np.full(
                (max_horizon_by_origin[origin],),
                train[-1],
                dtype=np.float64,
            )
        else:
            forecast = _fit_auto_ets_and_forecast(
                train=train,
                horizon=max_horizon_by_origin[origin],
                seasonality=seasonality,
            )
        for row_idx in row_ids:
            horizon = int(h_test[row_idx])
            step = min(max(horizon - 1, 0), forecast.size - 1)
            predictions[row_idx] = float(forecast[step])
    return predictions


def _append_metric_rows(
    *,
    out_rows: list[dict[str, Any]],
    dataset: str,
    bundle: DatasetBundle,
    series_id: int,
    model: str,
    y_test: np.ndarray,
    h_test: np.ndarray,
    pred: np.ndarray,
    insample: np.ndarray,
) -> None:
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
        out_rows.append(
            {
                "dataset": dataset,
                "frequency": bundle.frequency,
                "seasonality": int(bundle.seasonality),
                "series_id": int(series_id),
                "model": model,
                "horizon": int(horizon),
                "n_points": int(np.sum(mask)),
                "rmse": float(metric_values["rmse"]),
                "smape": float(metric_values["smape"]),
                "mase": float(metric_values["mase"]),
                "status": "ok",
                "skip_reason": "",
            }
        )


def _evaluate_stat_baselines(
    *,
    cfg: ForecastBenchmarkConfig,
    suite: dict[str, DatasetBundle],
    series_ids_by_dataset: dict[str, set[int]],
    stat_models: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    required_lag = cfg.protocol.required_lag

    for dataset_name, bundle in suite.items():
        allowed_ids = series_ids_by_dataset.get(dataset_name)
        if bundle.skipped:
            for model_name in stat_models:
                rows.append(
                    _skip_row(
                        dataset=dataset_name,
                        frequency=bundle.frequency,
                        seasonality=bundle.seasonality,
                        series_id=-1,
                        model=model_name,
                        reason=bundle.skip_reason or "dataset skipped",
                    )
                )
            continue

        for series_id, raw_series in enumerate(bundle.series):
            if allowed_ids is not None and series_id not in allowed_ids:
                continue

            series = np.asarray(raw_series, dtype=np.float64)
            if series.ndim != 1:
                for model_name in stat_models:
                    rows.append(
                        _skip_row(
                            dataset=dataset_name,
                            frequency=bundle.frequency,
                            seasonality=bundle.seasonality,
                            series_id=series_id,
                            model=model_name,
                            reason=(
                                f"Series {series_id} has invalid shape {series.shape}"
                            ),
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
            except Exception as exc:
                for model_name in stat_models:
                    rows.append(
                        _skip_row(
                            dataset=dataset_name,
                            frequency=bundle.frequency,
                            seasonality=bundle.seasonality,
                            series_id=series_id,
                            model=model_name,
                            reason=f"Failed preprocessing series {series_id}: {exc}",
                        )
                    )
                continue

            split = int(indices.split_index)
            series_filled = _fill_nan_1d(series)
            y_all = series_filled[indices.t_idx + indices.h_idx]
            t_test = indices.t_idx[split:]
            h_test = indices.h_idx[split:]
            y_test = y_all[split:]
            max_train_origin = int(np.max(indices.t_idx[:split]))
            insample = _fill_nan_1d(series[: max_train_origin + 1])

            if NAIVE_LAST in stat_models:
                naive_pred = series_filled[t_test]
                _append_metric_rows(
                    out_rows=rows,
                    dataset=dataset_name,
                    bundle=bundle,
                    series_id=series_id,
                    model=NAIVE_LAST,
                    y_test=y_test,
                    h_test=h_test,
                    pred=naive_pred,
                    insample=insample,
                )

            if AUTO_ETS in stat_models:
                try:
                    auto_ets_pred = _auto_ets_predict(
                        series_filled=series_filled,
                        t_test=t_test,
                        h_test=h_test,
                        seasonality=bundle.seasonality,
                    )
                    _append_metric_rows(
                        out_rows=rows,
                        dataset=dataset_name,
                        bundle=bundle,
                        series_id=series_id,
                        model=AUTO_ETS,
                        y_test=y_test,
                        h_test=h_test,
                        pred=auto_ets_pred,
                        insample=insample,
                    )
                except Exception as exc:
                    rows.append(
                        _skip_row(
                            dataset=dataset_name,
                            frequency=bundle.frequency,
                            seasonality=bundle.seasonality,
                            series_id=series_id,
                            model=AUTO_ETS,
                            reason=f"auto_ets failed: {exc}",
                        )
                    )

    return pd.DataFrame(rows)


def _write_outputs(
    *,
    rows: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    metadata: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = _rows_path(output_dir, prefix)
    perf_path = _perf_path(output_dir, prefix)
    status_path = _status_path(output_dir, prefix)
    meta_path = _meta_path(output_dir, prefix)

    rows.to_csv(rows_path, index=False)
    _compute_per_dataset_perf(rows).to_csv(perf_path, index=False)
    _status_summary(rows).to_csv(status_path, index=False)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "rows": rows_path,
        "perf": perf_path,
        "status": status_path,
        "meta": meta_path,
    }


def _resolve_stat_models(requested: list[str] | None) -> tuple[str, ...]:
    if not requested:
        return (NAIVE_LAST,)
    dedup: list[str] = []
    seen: set[str] = set()
    for model in requested:
        if model not in STAT_MODEL_CHOICES:
            raise ValueError(
                f"Unsupported statistical model: {model}. "
                f"Supported: {STAT_MODEL_CHOICES}"
            )
        if model not in seen:
            seen.add(model)
            dedup.append(model)
    return tuple(dedup)


def _append_stats_from_existing(
    *,
    config_path: Path,
    cache_dir: Path,
    timeout: float,
    existing_rows_path: Path,
    output_prefix: str,
    stat_models: tuple[str, ...],
) -> dict[str, Path]:
    if AUTO_ETS in stat_models and ExponentialSmoothing is None:
        raise RuntimeError(
            "auto_ets requested but statsmodels is not installed. "
            "Install statsmodels or remove auto_ets from --stat_models."
        )
    if not existing_rows_path.exists():
        raise FileNotFoundError(f"Missing existing rows CSV: {existing_rows_path}")

    existing = _normalize_rows(pd.read_csv(existing_rows_path, dtype=str))
    if "dataset" not in existing.columns:
        raise ValueError("Existing rows file must contain 'dataset' column.")

    dataset_names = tuple(
        str(v) for v in existing["dataset"].dropna().unique().tolist()
    )
    if not dataset_names:
        raise ValueError("No datasets found in existing rows file.")

    max_series = None
    metadata_path = (
        existing_rows_path.parent
        / _meta_path(
            existing_rows_path.parent,
            BASE_PREFIX,
        ).name
    )
    if metadata_path.exists():
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            max_series = int(payload.get("max_series_per_dataset"))
        except Exception:
            max_series = None
    if max_series is None:
        max_sid = existing.loc[existing["series_id"] >= 0, "series_id"].max()
        if not np.isfinite(max_sid):
            raise ValueError(
                "Could not infer max series count from existing rows. "
                "Ensure series_id >= 0 exists."
            )
        max_series = int(max_sid + 1)

    cfg = _build_stats_cfg(
        config_path=config_path,
        cache_dir=cache_dir,
        dataset_names=dataset_names,
        max_series_per_dataset=max_series,
    )
    _prepare_missing_caches(cfg, timeout=timeout)
    suite = load_suite(cfg)

    series_ids_by_dataset = _infer_series_ids_by_dataset(existing)
    baseline_rows = _normalize_rows(
        _evaluate_stat_baselines(
            cfg=cfg,
            suite=suite,
            series_ids_by_dataset=series_ids_by_dataset,
            stat_models=stat_models,
        )
    )
    retained = existing.loc[~existing["model"].isin(stat_models)].copy()
    merged = (
        pd.concat([retained, baseline_rows], axis=0, ignore_index=True)
        .sort_values(["dataset", "series_id", "model", "horizon"], na_position="last")
        .reset_index(drop=True)
    )

    metadata = {
        "derived_from": str(existing_rows_path),
        "added_models": list(stat_models),
        "datasets": list(dataset_names),
        "cache_dir": str(cfg.datasets.cache_dir),
        "max_series_per_dataset": int(cfg.datasets.max_series_per_dataset),
    }
    outputs = _write_outputs(
        rows=merged,
        output_dir=existing_rows_path.parent,
        prefix=output_prefix,
        metadata=metadata,
    )
    print("\nStatus counts for added statistical baselines:")
    print(_status_summary(baseline_rows).to_string(index=False))
    return outputs


def _run_full(args: argparse.Namespace, repo_root: Path) -> None:
    config_path = _resolve_path(repo_root, args.config)
    output_dir = _resolve_path(repo_root, args.output_dir)
    cache_dir = _resolve_path(repo_root, args.cache_dir)

    cfg = _build_full_cfg(
        config_path=config_path,
        output_dir=output_dir,
        cache_dir=cache_dir,
        max_series_per_dataset=args.max_series_per_dataset,
    )
    token_source = _ensure_token_available(repo_root, cfg.models.nicl_api_key_env)

    print("Mode:", args.mode)
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
        cfg,
        suite=suite,
        adapters=_build_adapters(cfg),
        device="cpu",
    )
    metadata = {
        "datasets": list(cfg.datasets.dataset_names),
        "models": ["nanotabpfn_standard", "nicl_regression"],
        "cache_dir": str(cfg.datasets.cache_dir),
        "output_dir": str(cfg.output_dir),
        "max_series_per_dataset": int(cfg.datasets.max_series_per_dataset),
        "nicl_endpoint": cfg.models.nicl_regression_endpoint,
        "token_source": token_source,
    }
    outputs = _write_outputs(
        rows=rows,
        output_dir=Path(cfg.output_dir),
        prefix=BASE_PREFIX,
        metadata=metadata,
    )

    print("\nStatus counts:")
    print(_status_summary(rows).to_string(index=False))
    print("\nPer-dataset performance:")
    print(_compute_per_dataset_perf(rows).to_string(index=False))

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
    for path in outputs.values():
        print(path)

    if not args.append_stats:
        return

    stat_models = _resolve_stat_models(args.stat_models)
    stats_outputs = _append_stats_from_existing(
        config_path=config_path,
        cache_dir=cache_dir,
        timeout=args.timeout,
        existing_rows_path=outputs["rows"],
        output_prefix=args.output_prefix,
        stat_models=stat_models,
    )
    print("\nStat-augmented artifacts:")
    for path in stats_outputs.values():
        print(path)


def _run_append_stats(args: argparse.Namespace, repo_root: Path) -> None:
    config_path = _resolve_path(repo_root, args.config)
    cache_dir = _resolve_path(repo_root, args.cache_dir)
    output_dir = _resolve_path(repo_root, args.output_dir)
    default_rows = _rows_path(output_dir, BASE_PREFIX)
    existing_rows_path = (
        _resolve_path(repo_root, args.existing_rows)
        if args.existing_rows is not None
        else default_rows
    )
    stat_models = _resolve_stat_models(args.stat_models)

    print("Mode:", args.mode)
    print("Config path:", config_path)
    print("Cache dir:", cache_dir)
    print("Existing rows:", existing_rows_path)
    print("Stat models:", list(stat_models))

    outputs = _append_stats_from_existing(
        config_path=config_path,
        cache_dir=cache_dir,
        timeout=args.timeout,
        existing_rows_path=existing_rows_path,
        output_prefix=args.output_prefix,
        stat_models=stat_models,
    )
    print("\nArtifacts:")
    for path in outputs.values():
        print(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified benchmark runner for expensive models and cheap statistical "
            "baseline append."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("full", "append_stats"),
        default="full",
        help="`full`: run standard+NICL, `append_stats`: add stats baselines only.",
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
        help="Output directory for rows/summaries.",
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
        help="Cap series per dataset in full mode.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Dataset cache-prep timeout (seconds).",
    )
    parser.add_argument(
        "--append_stats",
        action="store_true",
        help="In full mode, append statistical baselines after expensive run.",
    )
    parser.add_argument(
        "--stat_models",
        nargs="+",
        choices=STAT_MODEL_CHOICES,
        default=None,
        help="Statistical baselines to append (default: naive_last).",
    )
    parser.add_argument(
        "--existing_rows",
        type=Path,
        default=None,
        help=(
            "In append_stats mode, source regression rows CSV. "
            "Defaults to output_dir/regression_rows_standard_vs_nicl.csv."
        ),
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="standard_vs_nicl_plus_stats",
        help="Filename prefix for stat-augmented outputs.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if args.mode == "full":
        _run_full(args, repo_root)
        return
    _run_append_stats(args, repo_root)


if __name__ == "__main__":
    main()
