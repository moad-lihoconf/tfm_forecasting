from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch

import pretrain_regression_dynscm_live as live_train
from pretrain_regression import _prepare_output_path
from tfmplayground.gcs_utils import path_for_read, upload_local_file_to_gcs
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors.dynscm.difficulty import ridge_holdout_predictions
from tfmplayground.priors.dynscm.research_profiles import (
    LiveSourceSpec,
    get_research_profile,
)
from tfmplayground.utils import get_default_device

FeatureNormalization = Literal["per_function_zscore", "none"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--research_profile", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--batchsize", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dynscm_workers", type=int, default=4)
    parser.add_argument("--dynscm_worker_blas_threads", type=int, default=1)
    return parser


def _feature_normalization(value: object) -> FeatureNormalization:
    if value not in {"per_function_zscore", "none"}:
        raise ValueError(f"Unsupported feature normalization: {value!r}.")
    return cast(FeatureNormalization, value)


def _suite_metrics(
    *,
    model: NanoTabPFNModel,
    loader,
    device: torch.device,
) -> dict[str, float | int | dict[str, dict[str, float | int]]]:
    model.eval()
    total_batches = 0
    metric_state = {
        "model": {
            "squared_error_sum": 0.0,
            "target_sum": 0.0,
            "target_sq_sum": 0.0,
            "target_count": 0,
            "skipped_batches": 0,
        },
        "context_mean": {
            "squared_error_sum": 0.0,
            "target_sum": 0.0,
            "target_sq_sum": 0.0,
            "target_count": 0,
            "skipped_batches": 0,
        },
        "ridge": {
            "squared_error_sum": 0.0,
            "target_sum": 0.0,
            "target_sq_sum": 0.0,
            "target_count": 0,
            "skipped_batches": 0,
        },
    }

    def _prediction_metrics(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[float, float, float, int, bool]:
        preds = predictions.to(torch.float32)
        target_values = targets.to(torch.float32)
        if mask is not None:
            if not torch.any(mask):
                return 0.0, 0.0, 0.0, 0, True
            preds = preds[mask]
            target_values = target_values[mask]
        if not torch.isfinite(preds).all() or not torch.isfinite(target_values).all():
            return 0.0, 0.0, 0.0, 0, True
        errors = preds - target_values
        return (
            float(torch.sum(errors * errors).item()),
            float(torch.sum(target_values).item()),
            float(torch.sum(target_values * target_values).item()),
            int(target_values.numel()),
            False,
        )

    def _evaluate_subset(
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        target_y_batch: torch.Tensor,
        target_mask_batch: torch.Tensor | None,
        num_datapoints_batch: int | torch.Tensor | None,
        eval_pos: int,
    ) -> dict[str, tuple[float, float, float, int, bool]]:
        if torch.is_tensor(num_datapoints_batch):
            max_num_datapoints = int(
                cast(torch.Tensor, num_datapoints_batch).max().item()
            )
        elif isinstance(num_datapoints_batch, int):
            max_num_datapoints = num_datapoints_batch
        elif num_datapoints_batch is not None:
            max_num_datapoints = int(num_datapoints_batch)
        else:
            max_num_datapoints = x_batch.shape[1]
        x_batch = x_batch[:, :max_num_datapoints]
        y_batch = y_batch[:, :max_num_datapoints]
        target_y_batch = target_y_batch[:, :max_num_datapoints]
        mask = None
        if target_mask_batch is not None:
            target_mask_batch = target_mask_batch[:, :max_num_datapoints]
            mask = target_mask_batch[:, eval_pos:]
        targets = target_y_batch[:, eval_pos:]
        outputs = model((x_batch, y_batch[:, :eval_pos]), single_eval_pos=eval_pos)
        if outputs.ndim == targets.ndim + 1 and outputs.shape[-1] == 1:
            outputs = outputs.squeeze(-1)
        context = y_batch[:, :eval_pos].to(torch.float32)
        if context.shape[1] > 0:
            context_mean = context.mean(dim=1, keepdim=True).expand_as(targets)
        else:
            context_mean = torch.zeros_like(targets, dtype=torch.float32)

        ridge_predictions = torch.zeros_like(targets, dtype=torch.float32)
        x_cpu = x_batch.detach().to("cpu", dtype=torch.float32)
        y_cpu = target_y_batch.detach().to("cpu", dtype=torch.float32)
        if torch.is_tensor(num_datapoints_batch):
            row_lengths = (
                num_datapoints_batch.detach().to("cpu", dtype=torch.long).tolist()
            )
        else:
            row_lengths = [max_num_datapoints] * x_batch.shape[0]
        for row_idx, row_total in enumerate(row_lengths):
            row_total = int(max(0, min(int(row_total), max_num_datapoints)))
            if row_total <= eval_pos:
                continue
            preds_np = ridge_holdout_predictions(
                x_cpu[row_idx, :row_total].numpy(),
                y_cpu[row_idx, :row_total].numpy(),
                n_train=eval_pos,
            )
            if preds_np.size == 0:
                continue
            ridge_predictions[row_idx, : preds_np.shape[0]] = torch.from_numpy(
                np.asarray(preds_np, dtype=np.float32)
            ).to(device)

        return {
            "model": _prediction_metrics(outputs, targets, mask),
            "context_mean": _prediction_metrics(context_mean, targets, mask),
            "ridge": _prediction_metrics(ridge_predictions, targets, mask),
        }

    def _merge_metrics(
        state: dict[str, float | int],
        metrics: tuple[float, float, float, int, bool],
    ) -> None:
        sse, total, sq_total, count, skipped = metrics
        state["squared_error_sum"] = float(state["squared_error_sum"]) + sse
        state["target_sum"] = float(state["target_sum"]) + total
        state["target_sq_sum"] = float(state["target_sq_sum"]) + sq_total
        state["target_count"] = int(state["target_count"]) + count
        if skipped or count == 0:
            state["skipped_batches"] = int(state["skipped_batches"]) + 1

    with torch.no_grad():
        for batch in loader:
            total_batches += 1
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            target_y = batch["target_y"].to(device)
            target_mask = batch.get("target_mask")
            if target_mask is not None:
                target_mask = target_mask.to(device=device, dtype=torch.bool)
            num_datapoints = batch.get("num_datapoints")
            single_eval_pos = batch["single_eval_pos"]
            batch_sse = 0.0
            batch_sum = 0.0
            batch_sq_sum = 0.0
            batch_count = 0
            batch_skipped = False
            if isinstance(single_eval_pos, int):
                subset_metrics = _evaluate_subset(
                    x,
                    y,
                    target_y,
                    target_mask,
                    num_datapoints,
                    int(single_eval_pos),
                )
                for baseline_name, baseline_metrics in subset_metrics.items():
                    _merge_metrics(metric_state[baseline_name], baseline_metrics)
            elif torch.is_tensor(single_eval_pos):
                eval_positions = single_eval_pos.to(device=device, dtype=torch.long)
                for eval_pos in torch.unique(eval_positions).tolist():
                    row_mask = eval_positions == int(eval_pos)
                    sub_num_datapoints = (
                        num_datapoints[row_mask]
                        if torch.is_tensor(num_datapoints)
                        else num_datapoints
                    )
                    subset_metrics = _evaluate_subset(
                        x[row_mask],
                        y[row_mask],
                        target_y[row_mask],
                        target_mask[row_mask] if target_mask is not None else None,
                        sub_num_datapoints,
                        int(eval_pos),
                    )
                    for baseline_name, baseline_metrics in subset_metrics.items():
                        _merge_metrics(metric_state[baseline_name], baseline_metrics)
            else:
                raise ValueError("Unsupported single_eval_pos payload.")

    def _finalize_metrics(state: dict[str, float | int]) -> dict[str, float | int]:
        target_count = int(state["target_count"])
        loss = float("nan")
        rmse = float("nan")
        nrmse = float("nan")
        target_std = float("nan")
        if target_count > 0:
            loss = float(state["squared_error_sum"]) / target_count
            rmse = math.sqrt(loss) if loss >= 0.0 else float("nan")
            mean = float(state["target_sum"]) / target_count
            variance = max(
                (float(state["target_sq_sum"]) / target_count) - (mean * mean),
                0.0,
            )
            target_std = math.sqrt(variance)
            if math.isfinite(rmse) and target_std > 0.0:
                nrmse = rmse / target_std
        return {
            "loss": float(loss),
            "rmse": float(rmse),
            "nrmse": float(nrmse),
            "skipped_fraction": (
                float(int(state["skipped_batches"]) / total_batches)
                if total_batches > 0
                else 0.0
            ),
            "num_targets": int(target_count),
            "target_std": float(target_std),
        }

    model_metrics = _finalize_metrics(metric_state["model"])
    return {
        **model_metrics,
        "baselines": {
            "context_mean": _finalize_metrics(metric_state["context_mean"]),
            "ridge": _finalize_metrics(metric_state["ridge"]),
        },
    }


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    profile = get_research_profile(args.research_profile)
    device = torch.device(args.device or get_default_device())
    local_temp_dirs: list[Path] = []
    try:
        local_output_json, gcs_output_json = _prepare_output_path(
            args.output_json,
            local_temp_dirs,
        )
        checkpoint = torch.load(path_for_read(args.checkpoint_path), map_location="cpu")
        architecture = checkpoint["architecture"]
        model = NanoTabPFNModel(
            num_attention_heads=int(architecture["num_attention_heads"]),
            embedding_size=int(architecture["embedding_size"]),
            mlp_hidden_size=int(architecture["mlp_hidden_size"]),
            num_layers=int(architecture["num_layers"]),
            num_outputs=int(architecture["num_outputs"]),
            dropout=float(architecture.get("dropout", 0.0)),
            feature_normalization=_feature_normalization(
                architecture.get("feature_normalization", "per_function_zscore")
            ),
            debug_output_clamp=architecture.get("debug_output_clamp"),
        )
        model.load_state_dict(checkpoint["model"])
        model = model.to(device)

        results = {
            "checkpoint_path": args.checkpoint_path,
            "research_profile": profile.name,
            "device": str(device),
            "suites": {},
        }
        eval_batch_size = (
            int(args.batchsize)
            if args.batchsize is not None
            else int(profile.training_budget.batch_size)
        )
        for suite in profile.eval_suites:
            suite_steps = (
                int(args.eval_steps) if args.eval_steps is not None else suite.steps
            )
            print(
                "[synthetic-eval] suite="
                f"{suite.name} start steps={suite_steps} batch_size={eval_batch_size} "
                f"workers={args.dynscm_workers}",
                flush=True,
            )
            loader = live_train._build_prior_loader(
                source=LiveSourceSpec(kind="single", cfg=suite.cfg),
                num_steps=suite_steps,
                batch_size=eval_batch_size,
                num_datapoints_max=profile.max_seq_len,
                num_features=profile.max_features,
                device=device,
                seed=suite.seed,
                workers=args.dynscm_workers,
                worker_blas_threads=args.dynscm_worker_blas_threads,
                total_train_batches=max(1, suite_steps),
            )
            try:
                suite_metrics = {
                    **_suite_metrics(model=model, loader=loader, device=device),
                    "seed": int(suite.seed),
                    "steps": int(suite_steps),
                }
                results["suites"][suite.name] = suite_metrics
                print(
                    "[synthetic-eval] suite="
                    f"{suite.name} done loss={suite_metrics['loss']:.6f} "
                    f"nrmse={suite_metrics['nrmse']:.6f} "
                    f"context_mean_nrmse={suite_metrics['baselines']['context_mean']['nrmse']:.6f} "
                    f"ridge_nrmse={suite_metrics['baselines']['ridge']['nrmse']:.6f} "
                    f"skipped_fraction={suite_metrics['skipped_fraction']:.6f}",
                    flush=True,
                )
            finally:
                loader.close()

        Path(local_output_json).write_text(
            json.dumps(results, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if gcs_output_json is not None:
            upload_local_file_to_gcs(local_output_json, gcs_output_json)
        print(json.dumps(results, indent=2, sort_keys=True))
    finally:
        for temp_dir in local_temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
