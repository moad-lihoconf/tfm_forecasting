from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import cast

import h5py
import torch
from pfns.bar_distribution import FullSupportBarDistribution
from sklearn.metrics import r2_score
from torch import nn

from tfmplayground.callbacks import ConsoleLoggerCallback
from tfmplayground.evaluation import TOY_TASKS_REGRESSION, get_openml_predictions
from tfmplayground.gcs_utils import is_gcs_uri, path_for_read, upload_local_file_to_gcs
from tfmplayground.interface import NanoTabPFNRegressor
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors import PriorDumpDataLoader
from tfmplayground.priors.audit import audit_prior_dump, integrity_errors
from tfmplayground.train import train
from tfmplayground.utils import (
    get_default_device,
    make_global_bucket_edges,
    set_randomness_seed,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--priordump",
        type=str,
        default="50x3_1280k_regression.h5",
        help="path to the prior dump (local or gs://)",
    )
    parser.add_argument(
        "--saveweights",
        type=str,
        default="nanotabpfn_weights.pth",
        help="path to save the trained model to (local or gs://)",
    )
    parser.add_argument(
        "--savebuckets",
        type=str,
        default="nanotabpfn_buckets.pth",
        help="path to save the bucket edges to (local or gs://)",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=6,
        help="number of attention heads",
    )
    parser.add_argument(
        "--embeddingsize",
        type=int,
        default=192,
        help="the size of the embeddings used for the cells",
    )
    parser.add_argument(
        "--hiddensize",
        type=int,
        default=768,
        help="size of the hidden layer of the mlps",
    )
    parser.add_argument(
        "--layers", type=int, default=6, help="number of transformer layers"
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default=1,
        help="batch size used during training (before gradient accumulation)",
    )
    parser.add_argument(
        "--accumulate",
        type=int,
        default=1,
        help="number of gradients to accumulate before updating the weights",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw_schedulefree",
        choices=["adamw_schedulefree", "adamw"],
        help="optimizer used for training",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="number of steps that constitute one epoch (important for lr scheduler)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10000, help="number of epochs to train for"
    )
    parser.add_argument(
        "--loadcheckpoint",
        type=str,
        default=None,
        help="checkpoint from which to continue training (local or gs://)",
    )
    parser.add_argument(
        "--warm_start",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "load model weights from --loadcheckpoint but reset optimizer state, "
            "epoch counters, data offset, and early-stopping state"
        ),
    )
    parser.add_argument(
        "--n_buckets",
        type=int,
        default=100,
        help="number of buckets for the data loader",
    )
    parser.add_argument(
        "--regression_loss",
        type=str,
        default="bar",
        choices=["bar", "mse", "huber"],
        help="regression objective used during training",
    )
    parser.add_argument(
        "--huber_delta",
        type=float,
        default=1.0,
        help="delta parameter for Huber regression loss",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="fraction of prior functions used for validation (0,1)",
    )
    parser.add_argument(
        "--train_subset_size",
        type=int,
        default=None,
        help=(
            "optional deterministic cap on the number of training functions "
            "used after the train/validation split"
        ),
    )
    parser.add_argument(
        "--val_subset_size",
        type=int,
        default=None,
        help=(
            "optional deterministic cap on the number of validation functions "
            "used after the train/validation split"
        ),
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=2402,
        help="seed used for deterministic train/validation partitioning",
    )
    parser.add_argument(
        "--early_stopping_metric",
        type=str,
        default="val_loss",
        choices=["val_loss", "val_rmse"],
        help="validation metric used for early stopping and best checkpoint selection",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="number of validation evaluations without improvement before stopping",
    )
    parser.add_argument(
        "--early_stopping_min_delta",
        type=float,
        default=1e-4,
        help="minimum improvement required to reset patience",
    )
    parser.add_argument(
        "--eval_every_epochs",
        type=int,
        default=1,
        help="run validation every N epochs",
    )
    parser.add_argument(
        "--val_steps",
        type=str,
        default="auto_full",
        help=(
            "validation steps per evaluation. Use 'auto_full' to cover the full "
            "validation partition once per evaluation."
        ),
    )
    parser.add_argument(
        "--openml_eval_every_epochs",
        type=int,
        default=0,
        help=(
            "run OpenML toy regression evaluation every N epochs. "
            "Set to 0 to disable in-training OpenML evaluation."
        ),
    )
    parser.add_argument(
        "--strict_prior_integrity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "fail fast when prior integrity audit detects legacy or misaligned "
            "padding metadata."
        ),
    )
    parser.add_argument(
        "--save_best_weights",
        type=str,
        default=None,
        help="path for best model weights (defaults to <saveweights>.best.pth)",
    )
    parser.add_argument(
        "--max_train_hours",
        type=float,
        default=None,
        help="optional wall-clock limit in hours",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="weight decay value for optimizer",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="dropout rate for transformer and decoder blocks",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable CUDA mixed precision training when a CUDA device is available",
    )
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
        help="mixed precision dtype to use when --amp is enabled on CUDA",
    )
    parser.add_argument(
        "--loss_weighting",
        type=str,
        default="per_target",
        choices=["per_target", "per_function"],
        help="how to weight loss across variable numbers of supervised targets",
    )
    parser.add_argument(
        "--target_normalization",
        type=str,
        default="per_function_zscore",
        choices=["per_function_zscore", "per_function_clamped", "none"],
        help="normalization applied to regression targets before loss computation",
    )
    parser.add_argument(
        "--target_std_floor",
        type=float,
        default=1e-2,
        help=(
            "minimum standard deviation used for --target_normalization="
            "per_function_clamped"
        ),
    )
    parser.add_argument(
        "--min_train_target_std",
        type=float,
        default=0.0,
        help=(
            "skip per-function regression losses when the training-target standard "
            "deviation is below this threshold"
        ),
    )
    parser.add_argument(
        "--runname",
        type=str,
        default="nanoTFM",
        help="training run name, used for workdir checkpoints",
    )
    parser.add_argument(
        "--feature_normalization",
        type=str,
        default="per_function_zscore",
        choices=["per_function_zscore", "none"],
        help="normalization applied by the feature encoder",
    )
    parser.add_argument(
        "--debug_output_clamp",
        type=float,
        default=None,
        help="optional absolute clamp for scalar decoder outputs during diagnostics",
    )
    parser.add_argument(
        "--debug_train_trace_json",
        type=str,
        default=None,
        help="optional JSON path for structured per-batch training diagnostics",
    )
    parser.add_argument(
        "--debug_trace_first_n_batches",
        type=int,
        default=0,
        help="record structured diagnostics for the first N global train batches",
    )
    parser.add_argument(
        "--debug_trace_every_n_batches",
        type=int,
        default=0,
        help="record structured diagnostics every N global train batches",
    )
    parser.add_argument(
        "--checkpoint_gcs_dir",
        type=str,
        default=None,
        help=(
            "optional gs:// directory to upload latest/best checkpoints from "
            "workdir/<runname>/ after training"
        ),
    )
    return parser


def _prepare_output_path(path: str, temp_dirs: list[Path]) -> tuple[str, str | None]:
    if is_gcs_uri(path):
        tmp_dir = Path(tempfile.mkdtemp(prefix="tfmplayground_out_"))
        temp_dirs.append(tmp_dir)
        return str(tmp_dir / Path(path).name), path

    local_path = Path(path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    return str(local_path), None


def _upload_checkpoint_if_present(local_path: Path, gcs_uri: str) -> None:
    if local_path.exists():
        upload_local_file_to_gcs(local_path, gcs_uri)


def _resolve_val_steps(raw_value: str, *, val_count: int, batch_size: int) -> int:
    normalized = raw_value.strip().lower()
    if normalized == "auto_full":
        return max(1, math.ceil(val_count / batch_size))
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise ValueError(
            "--val_steps must be an integer >= 1 or the string 'auto_full'."
        ) from exc
    if parsed < 1:
        raise ValueError("--val_steps must be >= 1.")
    return parsed


def _resolve_amp_dtype(raw_value: str) -> torch.dtype:
    normalized = raw_value.strip().lower()
    if normalized == "float16":
        return torch.float16
    if normalized == "bfloat16":
        return torch.bfloat16
    raise ValueError("--amp_dtype must be one of: float16, bfloat16.")


def _audit_float(audit: dict[str, object], key: str) -> float:
    return float(cast(float | int, audit[key]))


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    set_randomness_seed(2402)

    if args.checkpoint_gcs_dir is not None and not is_gcs_uri(args.checkpoint_gcs_dir):
        raise ValueError("--checkpoint_gcs_dir must be a gs:// path.")

    if args.checkpoint_gcs_dir is not None:
        args.checkpoint_gcs_dir = args.checkpoint_gcs_dir.rstrip("/")

    local_temp_dirs: list[Path] = []
    try:
        local_priordump = path_for_read(args.priordump)
        local_debug_trace_path = None
        gcs_debug_trace_path = None
        if args.debug_train_trace_json is not None:
            local_debug_trace_path, gcs_debug_trace_path = _prepare_output_path(
                args.debug_train_trace_json,
                local_temp_dirs,
            )

        device = torch.device(get_default_device())
        ckpt = None
        if args.loadcheckpoint:
            ckpt_path = path_for_read(args.loadcheckpoint)
            ckpt = torch.load(ckpt_path)
            checkpoint_regression_loss = ckpt.get(
                "regression_loss",
                ckpt.get("training", {}).get("regression_loss", "bar"),
            )
            if checkpoint_regression_loss != args.regression_loss:
                raise ValueError(
                    "Checkpoint regression_loss does not match requested "
                    "--regression_loss."
                )
        amp_dtype = _resolve_amp_dtype(args.amp_dtype)
        use_amp = bool(args.amp and device.type == "cuda")
        print(
            f"Mixed precision {'enabled' if use_amp else 'disabled'}"
            + (
                f" (dtype={args.amp_dtype})"
                if use_amp
                else f" (requested={args.amp}, device={device.type})"
            ),
            flush=True,
        )

        if not 0.0 < args.val_split < 1.0:
            raise ValueError("--val_split must be in (0, 1).")
        if args.eval_every_epochs < 1:
            raise ValueError("--eval_every_epochs must be >= 1.")
        if args.openml_eval_every_epochs < 0:
            raise ValueError("--openml_eval_every_epochs must be >= 0.")
        if args.early_stopping_patience < 1:
            raise ValueError("--early_stopping_patience must be >= 1.")
        if args.dropout < 0.0 or args.dropout >= 1.0:
            raise ValueError("--dropout must be in [0, 1).")
        if args.huber_delta <= 0.0:
            raise ValueError("--huber_delta must be > 0.")
        if args.warm_start and args.loadcheckpoint is None:
            raise ValueError("--warm_start requires --loadcheckpoint.")
        if args.train_subset_size is not None and args.train_subset_size < 1:
            raise ValueError("--train_subset_size must be >= 1 when provided.")
        if args.val_subset_size is not None and args.val_subset_size < 1:
            raise ValueError("--val_subset_size must be >= 1 when provided.")
        if args.target_std_floor <= 0.0:
            raise ValueError("--target_std_floor must be > 0.")
        if args.min_train_target_std < 0.0:
            raise ValueError("--min_train_target_std must be >= 0.")
        if args.debug_output_clamp is not None and args.debug_output_clamp <= 0.0:
            raise ValueError("--debug_output_clamp must be > 0 when provided.")
        if args.debug_trace_first_n_batches < 0:
            raise ValueError("--debug_trace_first_n_batches must be >= 0.")
        if args.debug_trace_every_n_batches < 0:
            raise ValueError("--debug_trace_every_n_batches must be >= 0.")
        if args.regression_loss != "bar" and args.openml_eval_every_epochs > 0:
            raise ValueError(
                "--openml_eval_every_epochs requires --regression_loss=bar."
            )
        if ckpt:
            checkpoint_target_normalization = ckpt.get(
                "target_normalization",
                ckpt.get(
                    "training",
                    {},
                ).get("target_normalization", "per_function_zscore"),
            )
            if checkpoint_target_normalization != args.target_normalization:
                raise ValueError(
                    "Checkpoint target_normalization does not match requested "
                    "--target_normalization."
                )
            checkpoint_feature_normalization = ckpt.get(
                "architecture",
                {},
            ).get("feature_normalization", "per_function_zscore")
            if checkpoint_feature_normalization != args.feature_normalization:
                raise ValueError(
                    "Checkpoint feature_normalization does not match requested "
                    "--feature_normalization."
                )

        prior_audit = audit_prior_dump(local_priordump, sample_limit=4096)
        audit_failures = integrity_errors(prior_audit)
        feature_truncation_fraction = float(
            cast(
                float | int,
                prior_audit.get("feature_truncation_fraction", float("nan")),
            )
        )
        audit_summary = {
            "has_num_datapoints_dataset": bool(
                prior_audit["has_num_datapoints_dataset"]
            ),
            "has_variant_family_metadata": bool(
                prior_audit.get("has_variant_family_metadata", False)
            ),
            "inferred_padded_target_fraction": _audit_float(
                prior_audit,
                "inferred_padded_target_fraction",
            ),
            "inferred_num_datapoints_mismatch_fraction": _audit_float(
                prior_audit,
                "inferred_num_datapoints_mismatch_fraction",
            ),
            "feature_budget_saturation_fraction": _audit_float(
                prior_audit,
                "feature_budget_saturation_fraction",
            ),
            "feature_truncation_fraction": feature_truncation_fraction,
            "train_y_std_mean": _audit_float(prior_audit, "train_y_std_mean"),
            "target_y_std_mean": _audit_float(prior_audit, "target_y_std_mean"),
            "family_entropies": prior_audit.get("family_entropies", {}),
        }
        print(
            f"Prior audit summary: {json.dumps(audit_summary, sort_keys=True)}",
            flush=True,
        )
        if audit_failures:
            details = "\n".join(f"- {issue}" for issue in audit_failures)
            message = f"Prior integrity audit failed:\n{details}"
            if args.strict_prior_integrity:
                raise ValueError(message)
            print(
                f"WARNING: {message}",
                flush=True,
            )

        with h5py.File(local_priordump, "r") as f:
            total_functions = int(f["X"].shape[0])
        if total_functions < 2:
            raise ValueError(
                "Prior dump must contain at least 2 functions for train/val split."
            )

        val_count = int(total_functions * args.val_split)
        if val_count < 1 or val_count >= total_functions:
            raise ValueError(
                "Computed validation split is empty or full. "
                f"total={total_functions}, val_split={args.val_split}"
            )
        split_generator = torch.Generator(device="cpu")
        split_generator.manual_seed(args.split_seed)
        perm = torch.randperm(total_functions, generator=split_generator, device="cpu")
        val_indices = perm[:val_count].tolist()
        train_indices = perm[val_count:].tolist()

        if args.train_subset_size is not None:
            if args.train_subset_size > len(train_indices):
                raise ValueError(
                    "--train_subset_size exceeds the available training split size. "
                    f"train_subset_size={args.train_subset_size}, "
                    f"train={len(train_indices)}"
                )
            train_indices = train_indices[: args.train_subset_size]
        if args.val_subset_size is not None:
            if args.val_subset_size > len(val_indices):
                raise ValueError(
                    "--val_subset_size exceeds the available validation split size. "
                    f"val_subset_size={args.val_subset_size}, "
                    f"val={len(val_indices)}"
                )
            val_indices = val_indices[: args.val_subset_size]

        if len(train_indices) < args.batchsize or len(val_indices) < args.batchsize:
            raise ValueError(
                "Train/validation partitions must each contain at least one"
                " full batch. "
                f"train={len(train_indices)}, val={len(val_indices)}, "
                f"batchsize={args.batchsize}"
            )

        resumed_epoch = int(ckpt["epoch"]) if (ckpt and not args.warm_start) else 0
        train_start = args.steps * args.batchsize * resumed_epoch
        val_steps = _resolve_val_steps(
            args.val_steps,
            val_count=len(val_indices),
            batch_size=args.batchsize,
        )
        prior = PriorDumpDataLoader(
            filename=local_priordump,
            num_steps=args.steps,
            batch_size=args.batchsize,
            device=device,
            starting_index=train_start,
            indices=train_indices,
        )
        val_prior = PriorDumpDataLoader(
            filename=local_priordump,
            num_steps=val_steps,
            batch_size=args.batchsize,
            device=device,
            starting_index=0,
            indices=val_indices,
        )

        model = NanoTabPFNModel(
            num_attention_heads=args.heads,
            embedding_size=args.embeddingsize,
            mlp_hidden_size=args.hiddensize,
            num_layers=args.layers,
            num_outputs=(args.n_buckets if args.regression_loss == "bar" else 1),
            dropout=(
                float(ckpt["architecture"].get("dropout", args.dropout))
                if ckpt
                else args.dropout
            ),
            feature_normalization=(
                str(
                    ckpt["architecture"].get(
                        "feature_normalization",
                        args.feature_normalization,
                    )
                )
                if ckpt
                else args.feature_normalization
            ),
            debug_output_clamp=(
                (
                    None
                    if ckpt["architecture"].get("debug_output_clamp") is None
                    else float(ckpt["architecture"]["debug_output_clamp"])
                )
                if ckpt
                else args.debug_output_clamp
            ),
        )

        bucket_artifact: dict[str, object] | torch.Tensor
        criterion: nn.Module
        dist: FullSupportBarDistribution | None = None
        if args.regression_loss == "bar":
            bucket_edges = make_global_bucket_edges(
                filename=local_priordump,
                n_buckets=args.n_buckets,
                device=device,
                indices=train_indices,
            )
            bucket_artifact = bucket_edges
            dist = FullSupportBarDistribution(bucket_edges)
            criterion = dist
        elif args.regression_loss == "mse":
            bucket_artifact = {
                "regression_loss": args.regression_loss,
                "bucket_edges": None,
            }
            criterion = nn.MSELoss(reduction="none")
        else:
            bucket_artifact = {
                "regression_loss": args.regression_loss,
                "bucket_edges": None,
                "huber_delta": float(args.huber_delta),
            }
            criterion = nn.HuberLoss(delta=args.huber_delta, reduction="none")

        local_savebuckets, gcs_savebuckets = _prepare_output_path(
            args.savebuckets,
            local_temp_dirs,
        )
        torch.save(bucket_artifact, local_savebuckets)
        if gcs_savebuckets is not None:
            upload_local_file_to_gcs(local_savebuckets, gcs_savebuckets)

        if ckpt:
            checkpoint_optimizer = ckpt.get("optimizer_name", args.optimizer)
            if checkpoint_optimizer != args.optimizer:
                raise ValueError(
                    "Checkpoint optimizer_name does not match requested --optimizer."
                )
            model.load_state_dict(ckpt["model"])
        train_ckpt = None if args.warm_start else ckpt

        class EvaluationLoggerCallback(ConsoleLoggerCallback):
            def __init__(self, tasks: list[int], every_epochs: int):
                self.tasks = tasks
                self.every_epochs = every_epochs

            def on_epoch_end(
                self,
                epoch: int,
                epoch_time: float,
                loss: float,
                model,
                **kwargs,
            ):
                if self.every_epochs <= 0 or epoch % self.every_epochs != 0:
                    return
                if dist is None:
                    return
                regressor = NanoTabPFNRegressor(model, dist, device)
                predictions = get_openml_predictions(model=regressor, tasks=self.tasks)
                scores = []
                for _dataset_name, (y_true, y_pred, _) in predictions.items():
                    scores.append(r2_score(y_true, y_pred))
                if not scores:
                    return
                avg_score = sum(scores) / len(scores)
                print(
                    f"epoch {epoch:5d} | time {epoch_time:5.2f}s"
                    f" | mean loss {loss:5.2f}"
                    f" | avg r2 score {avg_score:.3f}",
                    flush=True,
                )

        callbacks: list[ConsoleLoggerCallback] = []
        if args.openml_eval_every_epochs > 0:
            callbacks.append(
                EvaluationLoggerCallback(
                    TOY_TASKS_REGRESSION,
                    every_epochs=args.openml_eval_every_epochs,
                )
            )

        effective_samples_per_epoch = max(1, args.steps * args.batchsize)
        min_epochs_before_stop = max(
            1,
            math.ceil(len(train_indices) / effective_samples_per_epoch),
        )

        trained_model, train_info = train(
            model=model,
            prior=prior,
            val_prior=val_prior,
            criterion=criterion,
            epochs=args.epochs,
            accumulate_gradients=args.accumulate,
            lr=args.lr,
            device=device,
            callbacks=callbacks,
            ckpt=train_ckpt,
            eval_every_epochs=args.eval_every_epochs,
            max_train_seconds=(
                None
                if args.max_train_hours is None
                else int(args.max_train_hours * 3600)
            ),
            early_stopping={
                "metric": args.early_stopping_metric,
                "patience": args.early_stopping_patience,
                "min_delta": args.early_stopping_min_delta,
                "min_epochs_before_stop": min_epochs_before_stop,
            },
            weight_decay=args.weight_decay,
            run_name=args.runname,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            loss_weighting=args.loss_weighting,
            optimizer_name=args.optimizer,
            regression_loss_name=args.regression_loss,
            target_normalization=args.target_normalization,
            target_std_floor=args.target_std_floor,
            min_train_target_std=args.min_train_target_std,
            debug_trace_path=local_debug_trace_path,
            debug_trace_first_n_batches=args.debug_trace_first_n_batches,
            debug_trace_every_n_batches=args.debug_trace_every_n_batches,
        )
        if (
            gcs_debug_trace_path is not None
            and local_debug_trace_path is not None
            and Path(local_debug_trace_path).exists()
        ):
            upload_local_file_to_gcs(local_debug_trace_path, gcs_debug_trace_path)
        train_info["debug_trace_path"] = args.debug_train_trace_json
        train_info["warm_start"] = bool(args.warm_start)
        train_info["loadcheckpoint"] = args.loadcheckpoint

        trained_model = trained_model.to("cpu")
        checkpoint_payload = {
            "architecture": {
                "num_layers": int(trained_model.num_layers),
                "embedding_size": int(trained_model.embedding_size),
                "num_attention_heads": int(trained_model.num_attention_heads),
                "mlp_hidden_size": int(trained_model.mlp_hidden_size),
                "num_outputs": int(trained_model.num_outputs),
                "dropout": float(getattr(trained_model, "dropout", args.dropout)),
                "feature_normalization": str(
                    getattr(
                        trained_model,
                        "feature_normalization",
                        args.feature_normalization,
                    )
                ),
                "debug_output_clamp": getattr(
                    trained_model,
                    "debug_output_clamp",
                    args.debug_output_clamp,
                ),
            },
            "model": trained_model.state_dict(),
            "training": train_info,
            "optimizer_name": args.optimizer,
            "regression_loss": args.regression_loss,
            "split": {
                "total_functions": total_functions,
                "train_count": len(train_indices),
                "val_count": len(val_indices),
                "val_split": args.val_split,
                "split_seed": args.split_seed,
                "train_subset_size": args.train_subset_size,
                "val_subset_size": args.val_subset_size,
                "val_steps": val_steps,
            },
            "prior_audit": prior_audit,
        }

        local_saveweights, gcs_saveweights = _prepare_output_path(
            args.saveweights,
            local_temp_dirs,
        )
        torch.save(checkpoint_payload, local_saveweights)
        if gcs_saveweights is not None:
            upload_local_file_to_gcs(local_saveweights, gcs_saveweights)

        best_ckpt_path = Path(f"workdir/{args.runname}/best_checkpoint.pth")
        save_best_weights_path = (
            args.save_best_weights or f"{args.saveweights}.best.pth"
        )
        best_payload = checkpoint_payload
        try:
            best_state = torch.load(best_ckpt_path, map_location=torch.device("cpu"))
            best_payload = {
                "architecture": best_state["architecture"],
                "model": best_state["model"],
                "regression_loss": args.regression_loss,
                "training": {
                    "best_epoch": best_state.get(
                        "best_epoch",
                        train_info.get("best_epoch", 0),
                    ),
                    "best_metric": best_state.get(
                        "best_metric",
                        train_info.get("best_metric", float("inf")),
                    ),
                    "stop_reason": train_info.get("stop_reason", "completed"),
                    "loss_weighting": train_info.get(
                        "loss_weighting",
                        args.loss_weighting,
                    ),
                    "optimizer_name": train_info.get(
                        "optimizer_name",
                        args.optimizer,
                    ),
                    "regression_loss": train_info.get(
                        "regression_loss",
                        args.regression_loss,
                    ),
                    "target_normalization": train_info.get(
                        "target_normalization",
                        args.target_normalization,
                    ),
                    "target_std_floor": train_info.get(
                        "target_std_floor",
                        args.target_std_floor,
                    ),
                    "min_train_target_std": train_info.get(
                        "min_train_target_std",
                        args.min_train_target_std,
                    ),
                    "debug_trace_path": train_info.get(
                        "debug_trace_path",
                        args.debug_train_trace_json,
                    ),
                    "split_seed": args.split_seed,
                    "train_subset_size": args.train_subset_size,
                    "val_subset_size": args.val_subset_size,
                },
            }
        except FileNotFoundError:
            pass

        local_best_weights, gcs_best_weights = _prepare_output_path(
            save_best_weights_path,
            local_temp_dirs,
        )
        torch.save(best_payload, local_best_weights)
        if gcs_best_weights is not None:
            upload_local_file_to_gcs(local_best_weights, gcs_best_weights)

        if args.checkpoint_gcs_dir is not None:
            latest_checkpoint = Path(f"workdir/{args.runname}/latest_checkpoint.pth")
            best_checkpoint = Path(f"workdir/{args.runname}/best_checkpoint.pth")
            _upload_checkpoint_if_present(
                latest_checkpoint,
                f"{args.checkpoint_gcs_dir}/latest_checkpoint.pth",
            )
            _upload_checkpoint_if_present(
                best_checkpoint,
                f"{args.checkpoint_gcs_dir}/best_checkpoint.pth",
            )
    finally:
        for temp_dir in local_temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
