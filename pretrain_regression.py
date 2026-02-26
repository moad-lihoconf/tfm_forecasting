from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import h5py
import torch
from pfns.bar_distribution import FullSupportBarDistribution
from sklearn.metrics import r2_score

from tfmplayground.callbacks import ConsoleLoggerCallback
from tfmplayground.evaluation import TOY_TASKS_REGRESSION, get_openml_predictions
from tfmplayground.gcs_utils import is_gcs_uri, path_for_read, upload_local_file_to_gcs
from tfmplayground.interface import NanoTabPFNRegressor
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors import PriorDumpDataLoader
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
        "--n_buckets",
        type=int,
        default=100,
        help="number of buckets for the data loader",
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="fraction of prior functions used for validation (0,1)",
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
        "--runname",
        type=str,
        default="nanoTFM",
        help="training run name, used for workdir checkpoints",
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

        device = get_default_device()
        ckpt = None
        if args.loadcheckpoint:
            ckpt_path = path_for_read(args.loadcheckpoint)
            ckpt = torch.load(ckpt_path)

        if not 0.0 < args.val_split < 1.0:
            raise ValueError("--val_split must be in (0, 1).")
        if args.eval_every_epochs < 1:
            raise ValueError("--eval_every_epochs must be >= 1.")
        if args.early_stopping_patience < 1:
            raise ValueError("--early_stopping_patience must be >= 1.")
        if args.dropout < 0.0 or args.dropout >= 1.0:
            raise ValueError("--dropout must be in [0, 1).")

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
        perm = torch.randperm(total_functions, device="cpu")
        val_indices = perm[:val_count].tolist()
        train_indices = perm[val_count:].tolist()

        if len(train_indices) < args.batchsize or len(val_indices) < args.batchsize:
            raise ValueError(
                "Train/validation partitions must each contain at least one"
                " full batch. "
                f"train={len(train_indices)}, val={len(val_indices)}, "
                f"batchsize={args.batchsize}"
            )

        train_start = args.steps * (ckpt["epoch"] if ckpt else 0)
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
            num_steps=max(1, args.steps // 4),
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
            num_outputs=args.n_buckets,
            dropout=(
                float(ckpt["architecture"].get("dropout", args.dropout))
                if ckpt
                else args.dropout
            ),
        )

        bucket_edges = make_global_bucket_edges(
            filename=local_priordump,
            n_buckets=args.n_buckets,
            device=device,
        )

        local_savebuckets, gcs_savebuckets = _prepare_output_path(
            args.savebuckets,
            local_temp_dirs,
        )
        torch.save(bucket_edges, local_savebuckets)
        if gcs_savebuckets is not None:
            upload_local_file_to_gcs(local_savebuckets, gcs_savebuckets)

        if ckpt:
            model.load_state_dict(ckpt["model"])

        dist = FullSupportBarDistribution(bucket_edges)

        class EvaluationLoggerCallback(ConsoleLoggerCallback):
            def __init__(self, tasks):
                self.tasks = tasks

            def on_epoch_end(
                self,
                epoch: int,
                epoch_time: float,
                loss: float,
                model,
                **kwargs,
            ):
                regressor = NanoTabPFNRegressor(model, dist, device)
                predictions = get_openml_predictions(model=regressor, tasks=self.tasks)
                scores = []
                for _dataset_name, (y_true, y_pred, _) in predictions.items():
                    scores.append(r2_score(y_true, y_pred))
                avg_score = sum(scores) / len(scores)
                print(
                    f"epoch {epoch:5d} | time {epoch_time:5.2f}s"
                    f" | mean loss {loss:5.2f}"
                    f" | avg r2 score {avg_score:.3f}",
                    flush=True,
                )

        callbacks = [EvaluationLoggerCallback(TOY_TASKS_REGRESSION)]

        trained_model, train_info = train(
            model=model,
            prior=prior,
            val_prior=val_prior,
            criterion=dist,
            epochs=args.epochs,
            accumulate_gradients=args.accumulate,
            lr=args.lr,
            device=device,
            callbacks=callbacks,
            ckpt=ckpt,
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
            },
            weight_decay=args.weight_decay,
            run_name=args.runname,
        )

        trained_model = trained_model.to("cpu")
        checkpoint_payload = {
            "architecture": {
                "num_layers": int(trained_model.num_layers),
                "embedding_size": int(trained_model.embedding_size),
                "num_attention_heads": int(trained_model.num_attention_heads),
                "mlp_hidden_size": int(trained_model.mlp_hidden_size),
                "num_outputs": int(trained_model.num_outputs),
                "dropout": float(getattr(trained_model, "dropout", args.dropout)),
            },
            "model": trained_model.state_dict(),
            "training": train_info,
            "split": {
                "total_functions": total_functions,
                "train_count": len(train_indices),
                "val_count": len(val_indices),
                "val_split": args.val_split,
            },
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
