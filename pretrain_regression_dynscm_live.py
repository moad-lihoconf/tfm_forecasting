from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Literal, TypedDict, cast

import numpy as np
import torch
from torch import nn

from pretrain_regression import (
    _prepare_output_path,
    _resolve_amp_dtype,
    _upload_checkpoint_if_present,
)
from scripts.summarize_train_trace import build_summary
from tfmplayground.gcs_utils import is_gcs_uri, path_for_read, upload_local_file_to_gcs
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors import PriorDataLoader
from tfmplayground.priors.dynscm import make_get_batch_dynscm
from tfmplayground.priors.dynscm.research import (
    MixtureGetBatch,
    NamedBatchModeSampler,
    sample_batch_shared_family_overrides,
)
from tfmplayground.priors.dynscm.research_profiles import (
    DynSCMLiveResearchProfile,
    LiveSourceSpec,
    build_promotion_profile,
    get_research_profile,
    list_research_profiles,
)
from tfmplayground.train import train
from tfmplayground.utils import get_default_device, set_randomness_seed

FeatureNormalization = Literal["per_function_zscore", "none"]


class ResolvedBudget(TypedDict):
    epochs: int
    steps: int
    batchsize: int
    accumulate: int
    lr: float
    weight_decay: float
    dropout: float
    amp: bool
    amp_dtype: Literal["float16", "bfloat16"]
    eval_every_epochs: int
    val_steps: int
    early_stopping_metric: str
    early_stopping_patience: int
    early_stopping_min_delta: float
    loss_weighting: Literal["per_target", "per_function"]
    grad_clip_norm: float
    debug_trace_first_n_batches: int
    debug_trace_every_n_batches: int


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--research_profile",
        type=str,
        required=True,
        choices=list_research_profiles(),
        help="named live DynSCM research profile to train",
    )
    parser.add_argument("--saveweights", type=str, default="nanotabpfn_weights.pth")
    parser.add_argument("--savebuckets", type=str, default="nanotabpfn_buckets.pth")
    parser.add_argument("--save_best_weights", type=str, default=None)
    parser.add_argument("--runname", type=str, default="nanoTFM-live-dynscm")
    parser.add_argument(
        "--promote_to_full",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "replace target_cfg training source with full_cfg while retaining strategy"
        ),
    )
    parser.add_argument("--checkpoint_gcs_dir", type=str, default=None)
    parser.add_argument("--debug_train_trace_json", type=str, default=None)
    parser.add_argument("--debug_train_trace_summary_json", type=str, default=None)
    parser.add_argument("--run_config_json", type=str, default=None)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--embeddingsize", type=int, default=192)
    parser.add_argument("--hiddensize", type=int, default=768)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--batchsize", type=int, default=None)
    parser.add_argument("--accumulate", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw_schedulefree", "adamw"],
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--loadcheckpoint", type=str, default=None)
    parser.add_argument(
        "--warm_start",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--regression_loss",
        type=str,
        default="mse",
        choices=["mse", "huber"],
    )
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument(
        "--early_stopping_metric",
        type=str,
        default=None,
        choices=["val_loss", "val_rmse"],
    )
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--early_stopping_min_delta", type=float, default=None)
    parser.add_argument("--eval_every_epochs", type=int, default=None)
    parser.add_argument("--val_steps", type=int, default=None)
    parser.add_argument("--max_train_hours", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default=None,
        choices=["float16", "bfloat16"],
    )
    parser.add_argument(
        "--loss_weighting",
        type=str,
        default=None,
        choices=["per_target", "per_function"],
    )
    parser.add_argument("--grad_clip_norm", type=float, default=None)
    parser.add_argument(
        "--target_normalization",
        type=str,
        default=None,
        choices=["per_function_zscore", "per_function_clamped", "none"],
    )
    parser.add_argument("--target_std_floor", type=float, default=None)
    parser.add_argument("--min_train_target_std", type=float, default=1e-3)
    parser.add_argument(
        "--feature_normalization",
        type=str,
        default="per_function_zscore",
        choices=["per_function_zscore", "none"],
    )
    parser.add_argument("--debug_output_clamp", type=float, default=None)
    parser.add_argument("--debug_trace_first_n_batches", type=int, default=None)
    parser.add_argument("--debug_trace_every_n_batches", type=int, default=None)
    parser.add_argument("--dynscm_workers", type=int, default=4)
    parser.add_argument("--dynscm_worker_blas_threads", type=int, default=1)
    return parser


def _budget_value(args_value: object, default_value: object) -> object:
    return default_value if args_value is None else args_value


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float | str):
        return int(value)
    raise TypeError(f"Expected int-compatible value, got {type(value).__name__}.")


def _as_float(value: object) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError(f"Expected float-compatible value, got {type(value).__name__}.")


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    raise TypeError(f"Expected bool-compatible value, got {type(value).__name__}.")


def _as_str(value: object) -> str:
    if isinstance(value, str):
        return value
    raise TypeError(f"Expected string value, got {type(value).__name__}.")


def _feature_normalization(value: object) -> FeatureNormalization:
    normalization = _as_str(value)
    if normalization not in {"per_function_zscore", "none"}:
        raise ValueError(f"Unsupported feature normalization: {normalization!r}.")
    return cast(FeatureNormalization, normalization)


def _batch_shared_sampler(
    cfg,
    family_fields: tuple[str, ...],
):
    def _numpy_sampler(
        rng: np.random.Generator,
        batch_index: int,
    ) -> dict[str, object]:
        del batch_index
        return sample_batch_shared_family_overrides(
            cfg,
            rng,
            family_fields=family_fields,
        )

    return _numpy_sampler


def _family_sampler(source: LiveSourceSpec):
    cfg = source.cfg
    if cfg is None or not source.batch_shared_fields:
        return None
    return _batch_shared_sampler(cfg, source.batch_shared_fields)


def _build_get_batch(
    *,
    source: LiveSourceSpec,
    seed: int,
    device: torch.device,
    workers: int,
    worker_blas_threads: int,
    total_train_batches: int,
):
    if source.kind == "single":
        if source.cfg is None:
            raise ValueError("single source requires cfg.")
        return make_get_batch_dynscm(
            source.cfg,
            device=device,
            seed=seed,
            workers=workers,
            worker_blas_threads=worker_blas_threads,
            cfg_override_sampler=_family_sampler(source),
            sample_filter=source.sample_filter,
            max_sample_attempts_per_item=source.max_sample_attempts_per_item,
            share_system_within_batch=source.share_system_within_batch,
        )
    if source.kind == "mode_ladder":
        if source.cfg is None or source.schedule is None:
            raise ValueError("mode_ladder source requires cfg and schedule.")
        mode_sampler = NamedBatchModeSampler(
            base_cfg=source.cfg,
            modes=source.modes,
            schedule=source.schedule,
            total_batches=total_train_batches,
            family_fields=source.batch_shared_fields,
        )
        return make_get_batch_dynscm(
            source.cfg,
            device=device,
            seed=seed,
            workers=workers,
            worker_blas_threads=worker_blas_threads,
            cfg_override_sampler=mode_sampler,
            sample_filter=source.sample_filter,
            max_sample_attempts_per_item=source.max_sample_attempts_per_item,
            share_system_within_batch=source.share_system_within_batch,
        )
    if source.kind == "mixture":
        if source.schedule is None:
            raise ValueError("mixture source requires schedule.")
        child_functions = []
        child_names = []
        for index, (child_name, child_cfg) in enumerate(source.child_sources):
            child_names.append(child_name)
            child_functions.append(
                make_get_batch_dynscm(
                    child_cfg,
                    device=device,
                    seed=seed + (index * 1000),
                    workers=workers,
                    worker_blas_threads=worker_blas_threads,
                    cfg_override_sampler=(
                        None
                        if not source.batch_shared_fields
                        else _batch_shared_sampler(
                            child_cfg,
                            source.batch_shared_fields,
                        )
                    ),
                    sample_filter=source.sample_filter,
                    max_sample_attempts_per_item=source.max_sample_attempts_per_item,
                    share_system_within_batch=source.share_system_within_batch,
                )
            )
        return MixtureGetBatch(
            children=tuple(child_functions),
            child_names=tuple(child_names),
            schedule=source.schedule,
            seed=seed,
            total_batches=total_train_batches,
        )
    raise ValueError(f"Unsupported source kind: {source.kind!r}.")


def _build_prior_loader(
    *,
    source: LiveSourceSpec,
    num_steps: int,
    batch_size: int,
    num_datapoints_max: int,
    num_features: int,
    device: torch.device,
    seed: int,
    workers: int,
    worker_blas_threads: int,
    total_train_batches: int,
) -> PriorDataLoader:
    return PriorDataLoader(
        get_batch_function=_build_get_batch(
            source=source,
            seed=seed,
            device=device,
            workers=workers,
            worker_blas_threads=worker_blas_threads,
            total_train_batches=total_train_batches,
        ),
        num_steps=num_steps,
        batch_size=batch_size,
        num_datapoints_max=num_datapoints_max,
        num_features=num_features,
        device=device,
    )


def _run_config_payload(
    *,
    args: argparse.Namespace,
    profile: DynSCMLiveResearchProfile,
    resolved_budget: ResolvedBudget,
    loadcheckpoint: str | None,
    warm_start: bool,
    resolved_target_normalization: str,
    resolved_target_std_floor: float,
    resolved_debug_output_clamp: float | None,
) -> dict[str, object]:
    return {
        "research_profile": profile.name,
        "warm_start_checkpoint": profile.warm_start_checkpoint,
        "effective_loadcheckpoint": loadcheckpoint,
        "effective_warm_start": bool(warm_start),
        "train_seed": int(profile.train_seed),
        "val_seed": int(profile.val_seed),
        "max_seq_len": int(profile.max_seq_len),
        "max_features": int(profile.max_features),
        "training_budget": resolved_budget,
        "cli": {
            "optimizer": args.optimizer,
            "regression_loss": args.regression_loss,
            "target_normalization": resolved_target_normalization,
            "target_std_floor": float(resolved_target_std_floor),
            "min_train_target_std": float(args.min_train_target_std),
            "feature_normalization": args.feature_normalization,
            "debug_output_clamp": args.debug_output_clamp,
            "resolved_debug_output_clamp": resolved_debug_output_clamp,
            "grad_clip_norm": float(resolved_budget["grad_clip_norm"]),
            "dynscm_workers": int(args.dynscm_workers),
            "dynscm_worker_blas_threads": int(args.dynscm_worker_blas_threads),
            "warm_start": bool(warm_start),
            "promote_to_full": bool(args.promote_to_full),
        },
        "sources": {
            "train": {
                "kind": profile.train_source.kind,
                "cfg": (
                    None
                    if profile.train_source.cfg is None
                    else profile.train_source.cfg.to_dict()
                ),
                "sample_filter": (
                    None
                    if profile.train_source.sample_filter is None
                    else profile.train_source.sample_filter.to_payload()
                ),
                "batch_shared_fields": list(profile.train_source.batch_shared_fields),
                "share_system_within_batch": bool(
                    profile.train_source.share_system_within_batch
                ),
                "max_sample_attempts_per_item": int(
                    profile.train_source.max_sample_attempts_per_item
                ),
                "modes": [
                    {"name": mode.name, "cfg_overrides": dict(mode.cfg_overrides)}
                    for mode in profile.train_source.modes
                ],
                "child_sources": [
                    {"name": name, "cfg": cfg.to_dict()}
                    for name, cfg in profile.train_source.child_sources
                ],
            },
            "val": {
                "kind": profile.val_source.kind,
                "cfg": (
                    None
                    if profile.val_source.cfg is None
                    else profile.val_source.cfg.to_dict()
                ),
                "share_system_within_batch": bool(
                    profile.val_source.share_system_within_batch
                ),
            },
        },
        "eval_suites": [
            {"name": suite.name, "cfg": suite.cfg.to_dict(), "steps": int(suite.steps)}
            for suite in profile.eval_suites
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    set_randomness_seed(2402)
    profile = (
        build_promotion_profile(args.research_profile)
        if args.promote_to_full
        else get_research_profile(args.research_profile)
    )
    budget = profile.training_budget
    resolved_target_normalization = (
        args.target_normalization
        if args.target_normalization is not None
        else profile.target_normalization
    )
    resolved_target_std_floor = (
        float(args.target_std_floor)
        if args.target_std_floor is not None
        else float(profile.target_std_floor)
    )

    resolved_budget: ResolvedBudget = {
        "epochs": _as_int(_budget_value(args.epochs, budget.epochs)),
        "steps": _as_int(_budget_value(args.steps, budget.steps)),
        "batchsize": _as_int(_budget_value(args.batchsize, budget.batch_size)),
        "accumulate": _as_int(_budget_value(args.accumulate, budget.accumulate)),
        "lr": _as_float(_budget_value(args.lr, budget.lr)),
        "weight_decay": _as_float(
            _budget_value(args.weight_decay, budget.weight_decay)
        ),
        "dropout": _as_float(_budget_value(args.dropout, budget.dropout)),
        "amp": _as_bool(_budget_value(args.amp, budget.amp)),
        "amp_dtype": cast(
            Literal["float16", "bfloat16"],
            _as_str(_budget_value(args.amp_dtype, budget.amp_dtype)),
        ),
        "eval_every_epochs": _as_int(
            _budget_value(args.eval_every_epochs, budget.eval_every_epochs)
        ),
        "val_steps": _as_int(_budget_value(args.val_steps, budget.val_steps)),
        "early_stopping_metric": _as_str(
            _budget_value(args.early_stopping_metric, budget.early_stopping_metric)
        ),
        "early_stopping_patience": _as_int(
            _budget_value(
                args.early_stopping_patience,
                budget.early_stopping_patience,
            )
        ),
        "early_stopping_min_delta": _as_float(
            _budget_value(
                args.early_stopping_min_delta,
                budget.early_stopping_min_delta,
            )
        ),
        "loss_weighting": cast(
            Literal["per_target", "per_function"],
            _as_str(_budget_value(args.loss_weighting, budget.loss_weighting)),
        ),
        "grad_clip_norm": _as_float(
            _budget_value(args.grad_clip_norm, budget.grad_clip_norm)
        ),
        "debug_trace_first_n_batches": _as_int(
            _budget_value(
                args.debug_trace_first_n_batches,
                budget.debug_trace_first_n_batches,
            )
        ),
        "debug_trace_every_n_batches": _as_int(
            _budget_value(
                args.debug_trace_every_n_batches,
                budget.debug_trace_every_n_batches,
            )
        ),
    }
    total_train_batches = resolved_budget["epochs"] * resolved_budget["steps"]
    if args.warm_start is None:
        effective_warm_start = args.loadcheckpoint is None
    else:
        effective_warm_start = bool(args.warm_start)
    effective_loadcheckpoint = (
        args.loadcheckpoint
        if args.loadcheckpoint is not None
        else (profile.warm_start_checkpoint if effective_warm_start else None)
    )
    if total_train_batches < 1:
        raise ValueError("epochs * steps must be >= 1.")
    if effective_warm_start and effective_loadcheckpoint is None:
        raise ValueError("--warm_start requires --loadcheckpoint.")
    if args.dynscm_workers < 1:
        raise ValueError("--dynscm_workers must be >= 1.")
    if args.dynscm_worker_blas_threads < 1:
        raise ValueError("--dynscm_worker_blas_threads must be >= 1.")
    if args.huber_delta <= 0.0:
        raise ValueError("--huber_delta must be > 0.")
    if resolved_target_std_floor <= 0.0:
        raise ValueError("--target_std_floor must be > 0.")
    if args.min_train_target_std < 0.0:
        raise ValueError("--min_train_target_std must be >= 0.")
    if resolved_budget["grad_clip_norm"] <= 0.0:
        raise ValueError("--grad_clip_norm must be > 0.")
    if args.checkpoint_gcs_dir is not None and not is_gcs_uri(args.checkpoint_gcs_dir):
        raise ValueError("--checkpoint_gcs_dir must be a gs:// path.")

    local_temp_dirs: list[Path] = []
    try:
        local_debug_trace_path = None
        gcs_debug_trace_path = None
        if args.debug_train_trace_json is not None:
            local_debug_trace_path, gcs_debug_trace_path = _prepare_output_path(
                args.debug_train_trace_json,
                local_temp_dirs,
            )
        local_debug_summary_path = None
        gcs_debug_summary_path = None
        if args.debug_train_trace_summary_json is not None:
            local_debug_summary_path, gcs_debug_summary_path = _prepare_output_path(
                args.debug_train_trace_summary_json,
                local_temp_dirs,
            )
        local_run_config_path = None
        gcs_run_config_path = None
        if args.run_config_json is not None:
            local_run_config_path, gcs_run_config_path = _prepare_output_path(
                args.run_config_json,
                local_temp_dirs,
            )

        device = torch.device(get_default_device())
        ckpt = None
        if effective_loadcheckpoint:
            ckpt = torch.load(
                path_for_read(effective_loadcheckpoint),
                map_location="cpu",
            )
            checkpoint_regression_loss = ckpt.get(
                "regression_loss",
                ckpt.get("training", {}).get("regression_loss", "mse"),
            )
            if checkpoint_regression_loss != args.regression_loss:
                raise ValueError(
                    "Checkpoint regression_loss does not match requested "
                    "--regression_loss."
                )
            checkpoint_target_normalization = ckpt.get(
                "target_normalization",
                ckpt.get("training", {}).get("target_normalization", "none"),
            )
            if (
                checkpoint_target_normalization != resolved_target_normalization
                and not effective_warm_start
            ):
                raise ValueError(
                    "Checkpoint target_normalization does not match requested "
                    "--target_normalization."
                )
            if (
                checkpoint_target_normalization != resolved_target_normalization
                and effective_warm_start
            ):
                print(
                    "Warm-start checkpoint target_normalization "
                    f"{checkpoint_target_normalization!r} differs from requested "
                    f"{resolved_target_normalization!r}; continuing because "
                    "--warm_start resets optimizer/training state.",
                    flush=True,
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
        cli_debug_output_clamp: float | None
        if args.debug_output_clamp is not None and args.debug_output_clamp < 0.0:
            cli_debug_output_clamp = None
        else:
            cli_debug_output_clamp = args.debug_output_clamp
        checkpoint_debug_output_clamp: float | None = None
        if ckpt is not None:
            raw_checkpoint_clamp = ckpt.get("architecture", {}).get(
                "debug_output_clamp",
                None,
            )
            checkpoint_debug_output_clamp = (
                None if raw_checkpoint_clamp is None else float(raw_checkpoint_clamp)
            )
        resolved_debug_output_clamp = (
            cli_debug_output_clamp
            if args.debug_output_clamp is not None
            else checkpoint_debug_output_clamp
        )

        amp_dtype = _resolve_amp_dtype(resolved_budget["amp_dtype"])
        use_amp = bool(resolved_budget["amp"] and device.type == "cuda")
        print(
            f"Mixed precision {'enabled' if use_amp else 'disabled'}"
            + (
                f" (dtype={resolved_budget['amp_dtype']})"
                if use_amp
                else f" (requested={resolved_budget['amp']}, device={device.type})"
            ),
            flush=True,
        )

        train_prior = _build_prior_loader(
            source=profile.train_source,
            num_steps=resolved_budget["steps"],
            batch_size=resolved_budget["batchsize"],
            num_datapoints_max=profile.max_seq_len,
            num_features=profile.max_features,
            device=device,
            seed=profile.train_seed,
            workers=args.dynscm_workers,
            worker_blas_threads=args.dynscm_worker_blas_threads,
            total_train_batches=total_train_batches,
        )
        val_prior = _build_prior_loader(
            source=profile.val_source,
            num_steps=resolved_budget["val_steps"],
            batch_size=resolved_budget["batchsize"],
            num_datapoints_max=profile.max_seq_len,
            num_features=profile.max_features,
            device=device,
            seed=profile.val_seed,
            workers=args.dynscm_workers,
            worker_blas_threads=args.dynscm_worker_blas_threads,
            total_train_batches=max(1, resolved_budget["val_steps"]),
        )

        model = NanoTabPFNModel(
            num_attention_heads=args.heads,
            embedding_size=args.embeddingsize,
            mlp_hidden_size=args.hiddensize,
            num_layers=args.layers,
            num_outputs=1,
            dropout=(
                float(ckpt["architecture"].get("dropout", resolved_budget["dropout"]))
                if ckpt
                else resolved_budget["dropout"]
            ),
            feature_normalization=(
                _feature_normalization(
                    ckpt["architecture"].get(
                        "feature_normalization",
                        args.feature_normalization,
                    )
                )
                if ckpt
                else _feature_normalization(args.feature_normalization)
            ),
            debug_output_clamp=resolved_debug_output_clamp,
        )
        if ckpt:
            checkpoint_optimizer = ckpt.get("optimizer_name", args.optimizer)
            if checkpoint_optimizer != args.optimizer:
                raise ValueError(
                    "Checkpoint optimizer_name does not match requested --optimizer."
                )
            model.load_state_dict(ckpt["model"])
        train_ckpt = None if effective_warm_start else ckpt

        if args.regression_loss == "mse":
            bucket_artifact: dict[str, object] = {
                "regression_loss": args.regression_loss,
                "bucket_edges": None,
            }
            criterion: nn.Module = nn.MSELoss(reduction="none")
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

        min_epochs_before_stop = max(1, 1)

        trained_model, train_info = train(
            model=model,
            prior=train_prior,
            val_prior=val_prior,
            criterion=criterion,
            epochs=resolved_budget["epochs"],
            accumulate_gradients=resolved_budget["accumulate"],
            lr=resolved_budget["lr"],
            device=device,
            callbacks=[],
            ckpt=train_ckpt,
            eval_every_epochs=resolved_budget["eval_every_epochs"],
            max_train_seconds=(
                None
                if args.max_train_hours is None
                else int(args.max_train_hours * 3600)
            ),
            early_stopping={
                "metric": resolved_budget["early_stopping_metric"],
                "patience": resolved_budget["early_stopping_patience"],
                "min_delta": resolved_budget["early_stopping_min_delta"],
                "min_epochs_before_stop": min_epochs_before_stop,
            },
            weight_decay=resolved_budget["weight_decay"],
            run_name=args.runname,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            loss_weighting=resolved_budget["loss_weighting"],
            optimizer_name=args.optimizer,
            regression_loss_name=args.regression_loss,
            target_normalization=resolved_target_normalization,
            target_std_floor=resolved_target_std_floor,
            min_train_target_std=args.min_train_target_std,
            grad_clip_norm=resolved_budget["grad_clip_norm"],
            debug_trace_path=local_debug_trace_path,
            debug_trace_first_n_batches=resolved_budget["debug_trace_first_n_batches"],
            debug_trace_every_n_batches=resolved_budget["debug_trace_every_n_batches"],
        )
        if (
            gcs_debug_trace_path is not None
            and local_debug_trace_path is not None
            and Path(local_debug_trace_path).exists()
        ):
            upload_local_file_to_gcs(local_debug_trace_path, gcs_debug_trace_path)
        if (
            local_debug_summary_path is not None
            and local_debug_trace_path is not None
            and Path(local_debug_trace_path).exists()
        ):
            with Path(local_debug_trace_path).open("r", encoding="utf-8") as handle:
                trace_payload = json.load(handle)
            summary = build_summary(trace_payload)
            Path(local_debug_summary_path).write_text(
                json.dumps(summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            if gcs_debug_summary_path is not None:
                upload_local_file_to_gcs(
                    local_debug_summary_path,
                    gcs_debug_summary_path,
                )

        if local_run_config_path is not None:
            Path(local_run_config_path).write_text(
                json.dumps(
                    _run_config_payload(
                        args=args,
                        profile=profile,
                        resolved_budget=resolved_budget,
                        loadcheckpoint=effective_loadcheckpoint,
                        warm_start=effective_warm_start,
                        resolved_target_normalization=resolved_target_normalization,
                        resolved_target_std_floor=resolved_target_std_floor,
                        resolved_debug_output_clamp=resolved_debug_output_clamp,
                    ),
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            if gcs_run_config_path is not None:
                upload_local_file_to_gcs(local_run_config_path, gcs_run_config_path)

        train_info["debug_trace_path"] = args.debug_train_trace_json
        train_info["debug_trace_summary_path"] = args.debug_train_trace_summary_json
        train_info["run_config_json"] = args.run_config_json
        train_info["research_profile"] = profile.name
        train_info["warm_start"] = bool(effective_warm_start)
        train_info["loadcheckpoint"] = effective_loadcheckpoint

        trained_model = trained_model.to("cpu")
        checkpoint_payload = {
            "architecture": {
                "num_layers": int(trained_model.num_layers),
                "embedding_size": int(trained_model.embedding_size),
                "num_attention_heads": int(trained_model.num_attention_heads),
                "mlp_hidden_size": int(trained_model.mlp_hidden_size),
                "num_outputs": int(trained_model.num_outputs),
                "dropout": float(getattr(trained_model, "dropout", 0.0)),
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
                    resolved_debug_output_clamp,
                ),
            },
            "model": trained_model.state_dict(),
            "training": train_info,
            "optimizer_name": args.optimizer,
            "regression_loss": args.regression_loss,
            "live_profile": _run_config_payload(
                args=args,
                profile=profile,
                resolved_budget=resolved_budget,
                loadcheckpoint=effective_loadcheckpoint,
                warm_start=effective_warm_start,
                resolved_target_normalization=resolved_target_normalization,
                resolved_target_std_floor=resolved_target_std_floor,
                resolved_debug_output_clamp=resolved_debug_output_clamp,
            ),
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
                        resolved_budget["loss_weighting"],
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
                        resolved_target_normalization,
                    ),
                    "target_std_floor": train_info.get(
                        "target_std_floor",
                        resolved_target_std_floor,
                    ),
                    "min_train_target_std": train_info.get(
                        "min_train_target_std",
                        args.min_train_target_std,
                    ),
                    "research_profile": profile.name,
                    "debug_trace_path": train_info.get(
                        "debug_trace_path",
                        args.debug_train_trace_json,
                    ),
                    "debug_trace_summary_path": train_info.get(
                        "debug_trace_summary_path",
                        args.debug_train_trace_summary_json,
                    ),
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
            checkpoint_dir = args.checkpoint_gcs_dir.rstrip("/")
            latest_checkpoint = Path(f"workdir/{args.runname}/latest_checkpoint.pth")
            best_checkpoint = Path(f"workdir/{args.runname}/best_checkpoint.pth")
            _upload_checkpoint_if_present(
                latest_checkpoint,
                f"{checkpoint_dir}/latest_checkpoint.pth",
            )
            _upload_checkpoint_if_present(
                best_checkpoint,
                f"{checkpoint_dir}/best_checkpoint.pth",
            )
    finally:
        for temp_dir in local_temp_dirs:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
