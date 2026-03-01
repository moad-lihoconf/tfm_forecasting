import json
import os
import time
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import schedulefree
import torch
from pfns.bar_distribution import FullSupportBarDistribution
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tfmplayground.callbacks import Callback
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.utils import get_default_device


@dataclass(frozen=True)
class _LossResult:
    loss: torch.Tensor
    weight: int
    debug: dict[str, Any] | None = None


def _tensor_stats(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        return {
            "min": float("nan"),
            "median": float("nan"),
            "mean": float("nan"),
            "max": float("nan"),
        }
    flattened = values.detach().to(torch.float32).reshape(-1)
    return {
        "min": float(flattened.min().item()),
        "median": float(flattened.median().item()),
        "mean": float(flattened.mean().item()),
        "max": float(flattened.max().item()),
    }


def _parameter_grad_norm(parameters) -> float | None:
    total = 0.0
    has_grad = False
    for parameter in parameters:
        grad = getattr(parameter, "grad", None)
        if grad is None:
            continue
        grad_norm = float(grad.detach().norm().item())
        total += grad_norm**2
        has_grad = True
    if not has_grad:
        return None
    return total**0.5


def _should_record_debug_trace(
    *,
    global_batch_index: int,
    first_n_batches: int,
    every_n_batches: int,
) -> bool:
    if first_n_batches > 0 and global_batch_index <= first_n_batches:
        return True
    return every_n_batches > 0 and global_batch_index % every_n_batches == 0


def _write_debug_trace(
    *,
    path: str,
    metadata: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "records": records}, handle, indent=2)


def _get_decoder_linear2(
    wrapped_model: NanoTabPFNModel | nn.DataParallel[NanoTabPFNModel],
) -> nn.Linear | None:
    model = (
        wrapped_model.module
        if isinstance(wrapped_model, nn.DataParallel)
        else wrapped_model
    )
    decoder = getattr(model, "decoder", None)
    if decoder is not None and hasattr(decoder, "linear2"):
        linear = decoder.linear2
        if isinstance(linear, nn.Linear):
            return linear
    for attr in ("linear2", "linear"):
        candidate = getattr(model, attr, None)
        if isinstance(candidate, nn.Linear):
            return candidate
    return None


def _build_optimizer(
    *,
    optimizer_name: Literal["adamw_schedulefree", "adamw"],
    parameters,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if optimizer_name == "adamw_schedulefree":
        return schedulefree.AdamWScheduleFree(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
        )
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
        )
    raise ValueError("optimizer_name must be one of {'adamw_schedulefree', 'adamw'}.")


def _set_optimizer_mode(optimizer: torch.optim.Optimizer, *, training: bool) -> None:
    mode_method = "train" if training else "eval"
    maybe_mode = getattr(optimizer, mode_method, None)
    if callable(maybe_mode):
        maybe_mode()


def _set_criterion_mode(
    criterion: nn.Module,
    *,
    training: bool,
) -> None:
    if isinstance(criterion, nn.Module):
        criterion.train(training)


def _is_scalar_regression_criterion(criterion: nn.Module) -> bool:
    return isinstance(criterion, nn.MSELoss | nn.HuberLoss)


def _compute_loss_result(
    wrapped_model: NanoTabPFNModel | nn.DataParallel[NanoTabPFNModel],
    criterion: nn.Module,
    full_data: dict[str, Any],
    device: torch.device,
    regression_task: bool,
    classification_task: bool,
    loss_weighting: Literal["per_target", "per_function"] = "per_target",
    target_normalization: Literal[
        "per_function_zscore", "per_function_clamped", "none"
    ] = "per_function_zscore",
    target_std_floor: float = 1e-2,
    min_train_target_std: float = 0.0,
) -> _LossResult:
    if loss_weighting not in {"per_target", "per_function"}:
        raise ValueError(
            "loss_weighting must be one of {'per_target', 'per_function'}."
        )
    if min_train_target_std < 0.0:
        raise ValueError("min_train_target_std must be >= 0.")
    if target_std_floor <= 0.0:
        raise ValueError("target_std_floor must be > 0.")
    if target_normalization not in {
        "per_function_zscore",
        "per_function_clamped",
        "none",
    }:
        raise ValueError(
            "target_normalization must be one of "
            "{'per_function_zscore', 'per_function_clamped', 'none'}."
        )
    single_eval_pos = full_data["single_eval_pos"]
    x = full_data["x"].to(device)
    y = full_data["y"].to(device)
    target_y = full_data["target_y"].to(device)
    target_mask_full = full_data.get("target_mask")
    num_datapoints_full = full_data.get("num_datapoints")
    if target_mask_full is not None:
        target_mask_full = target_mask_full.to(device=device, dtype=torch.bool)
        if target_mask_full.ndim != 2:
            raise ValueError("target_mask must have shape (batch_size, num_rows).")
        if target_mask_full.shape != y.shape:
            raise ValueError("target_mask shape must match y/target_y shape.")
    if torch.is_tensor(num_datapoints_full):
        num_datapoints_full = num_datapoints_full.to(device=device, dtype=torch.long)

    def _loss_for_fixed_eval_pos(
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        target_y_batch: torch.Tensor,
        eval_pos: int,
        target_mask_batch: torch.Tensor | None = None,
        num_datapoints_batch: int | torch.Tensor | None = None,
    ) -> _LossResult:
        debug: dict[str, Any] = {
            "batch_size": int(x_batch.shape[0]),
            "eval_pos": int(eval_pos),
            "filtered_low_std_functions": 0,
        }
        if torch.is_tensor(num_datapoints_batch):
            max_num_datapoints = int(num_datapoints_batch.max().item())
        elif num_datapoints_batch is not None:
            max_num_datapoints = int(num_datapoints_batch)
        else:
            max_num_datapoints = x_batch.shape[1]
        x_batch = x_batch[:, :max_num_datapoints]
        y_batch = y_batch[:, :max_num_datapoints]
        target_y_batch = target_y_batch[:, :max_num_datapoints]
        if target_mask_batch is not None:
            target_mask_batch = target_mask_batch[:, :max_num_datapoints]
        data = (x_batch, y_batch[:, :eval_pos])
        targets = target_y_batch[:, eval_pos:]
        mask = None if target_mask_batch is None else target_mask_batch[:, eval_pos:]

        if regression_task:
            y_mean = data[1].mean(dim=1, keepdim=True)
            y_std = data[1].std(dim=1, keepdim=True)
            debug["train_target_std"] = _tensor_stats(y_std.squeeze(1))
            if min_train_target_std > 0.0:
                valid = torch.isfinite(y_std.squeeze(1)) & (
                    y_std.squeeze(1) >= min_train_target_std
                )
                debug["filtered_low_std_functions"] = int((~valid).sum().item())
                if not torch.any(valid):
                    return _LossResult(
                        loss=torch.tensor(float("nan"), device=device),
                        weight=0,
                        debug=debug,
                    )
                if not torch.all(valid):
                    x_batch = x_batch[valid]
                    y_batch = y_batch[valid]
                    target_y_batch = target_y_batch[valid]
                    if target_mask_batch is not None:
                        target_mask_batch = target_mask_batch[valid]
                    if torch.is_tensor(num_datapoints_batch):
                        num_datapoints_batch = num_datapoints_batch[valid]
                    data = (x_batch, y_batch[:, :eval_pos])
                    targets = target_y_batch[:, eval_pos:]
                    mask = (
                        None
                        if target_mask_batch is None
                        else target_mask_batch[:, eval_pos:]
                    )
                    y_mean = data[1].mean(dim=1, keepdim=True)
                    y_std = data[1].std(dim=1, keepdim=True)
            if target_normalization == "per_function_zscore":
                y_scale = y_std + 1e-8
                data = (data[0], (data[1] - y_mean) / y_scale)
                targets = (targets - y_mean) / y_scale
            elif target_normalization == "per_function_clamped":
                y_scale = torch.clamp(y_std, min=target_std_floor)
                data = (data[0], (data[1] - y_mean) / y_scale)
                targets = (targets - y_mean) / y_scale
            else:
                y_scale = torch.ones_like(y_std)
            debug["normalized_target_abs"] = _tensor_stats(targets.abs())

        output = wrapped_model(data, single_eval_pos=eval_pos)
        if (
            regression_task
            and output.ndim == targets.ndim + 1
            and output.shape[-1] == 1
        ):
            output = output.squeeze(-1)
        debug["output_abs"] = _tensor_stats(output.abs())
        if classification_task:
            targets = targets.reshape((-1,)).to(torch.long)
            output = output.view(-1, output.shape[-1])
            if mask is not None:
                mask_flat = mask.reshape(-1)
                if not torch.any(mask_flat):
                    return _LossResult(
                        loss=torch.tensor(float("nan"), device=device),
                        weight=0,
                        debug=debug,
                    )
                targets = targets[mask_flat]
                output = output[mask_flat]

        if regression_task and _is_scalar_regression_criterion(criterion):
            losses = criterion(output.to(torch.float32), targets.to(torch.float32))
        else:
            losses = criterion(output, targets)
        valid_supervised_targets = (
            int(mask.sum().item()) if mask is not None else int(targets.numel())
        )
        debug["valid_supervised_targets"] = valid_supervised_targets
        if mask is None or classification_task:
            loss = losses.mean() if losses.ndim > 0 else losses
            if loss_weighting == "per_function":
                weight = x_batch.shape[0]
            else:
                weight = int(targets.numel())
            debug["loss"] = float(loss.detach().to(torch.float32).item())
            return _LossResult(loss=loss, weight=weight, debug=debug)
        if losses.ndim == 0:
            weight = x_batch.shape[0] if loss_weighting == "per_function" else 1
            debug["loss"] = float(losses.detach().to(torch.float32).item())
            return _LossResult(loss=losses, weight=weight, debug=debug)
        masked_losses = losses[mask]
        if masked_losses.numel() == 0:
            return _LossResult(
                loss=torch.tensor(float("nan"), device=device),
                weight=0,
                debug=debug,
            )
        weight = (
            x_batch.shape[0]
            if loss_weighting == "per_function"
            else int(mask.sum().item())
        )
        loss = masked_losses.mean()
        debug["loss"] = float(loss.detach().to(torch.float32).item())
        return _LossResult(loss=loss, weight=weight, debug=debug)

    if isinstance(single_eval_pos, int):
        return _loss_for_fixed_eval_pos(
            x,
            y,
            target_y,
            int(single_eval_pos),
            target_mask_full,
            num_datapoints_full,
        )

    if torch.is_tensor(single_eval_pos):
        if single_eval_pos.ndim == 0:
            return _loss_for_fixed_eval_pos(
                x,
                y,
                target_y,
                int(single_eval_pos.item()),
                target_mask_full,
                num_datapoints_full,
            )
        if single_eval_pos.ndim != 1:
            raise ValueError("single_eval_pos tensor must be scalar or 1D.")
        if single_eval_pos.shape[0] != x.shape[0]:
            raise ValueError(
                "single_eval_pos batch dimension does not match input batch size."
            )

        sep = single_eval_pos.to(device=x.device, dtype=torch.long)
        unique_sep = torch.unique(sep)
        weighted_loss = torch.tensor(0.0, device=device)
        total_weight = 0
        debug_children: list[tuple[int, dict[str, Any]]] = []
        for eval_pos in unique_sep.tolist():
            mask = sep == int(eval_pos)
            if not torch.any(mask):
                continue
            batch_loss = _loss_for_fixed_eval_pos(
                x[mask],
                y[mask],
                target_y[mask],
                int(eval_pos),
                target_mask_full[mask] if target_mask_full is not None else None,
                num_datapoints_full[mask]
                if torch.is_tensor(num_datapoints_full)
                else num_datapoints_full,
            )
            if batch_loss.weight <= 0 or torch.isnan(batch_loss.loss):
                continue
            weighted_loss = weighted_loss + batch_loss.loss * batch_loss.weight
            total_weight += batch_loss.weight
            if batch_loss.debug is not None:
                debug_children.append((batch_loss.weight, batch_loss.debug))
        if total_weight == 0:
            return _LossResult(
                loss=torch.tensor(float("nan"), device=device),
                weight=0,
            )
        debug_summary: dict[str, Any] | None = None
        if debug_children:
            debug_summary = {
                "batch_size": int(x.shape[0]),
                "eval_pos": [int(v) for v in unique_sep.tolist()],
                "filtered_low_std_functions": int(
                    sum(
                        int(child["filtered_low_std_functions"])
                        for _, child in debug_children
                    )
                ),
                "valid_supervised_targets": int(
                    sum(
                        int(child["valid_supervised_targets"])
                        for _, child in debug_children
                    )
                ),
                "loss": float(
                    (weighted_loss / total_weight).detach().to(torch.float32).item()
                ),
            }
            for key in ("train_target_std", "normalized_target_abs", "output_abs"):
                weighted_values = [
                    (weight, child[key])
                    for weight, child in debug_children
                    if key in child and isinstance(child[key], dict)
                ]
                if weighted_values:
                    for stat_name in ("min", "median", "mean", "max"):
                        weighted_value = (
                            sum(
                                float(value[stat_name]) * weight
                                for weight, value in weighted_values
                            )
                            / total_weight
                        )
                        debug_summary.setdefault(key, {})[stat_name] = float(
                            weighted_value
                        )
        return _LossResult(
            loss=weighted_loss / total_weight,
            weight=total_weight,
            debug=debug_summary,
        )

    raise TypeError("single_eval_pos must be int or torch.Tensor.")


def _compute_loss(
    wrapped_model: NanoTabPFNModel | nn.DataParallel[NanoTabPFNModel],
    criterion: nn.Module,
    full_data: dict[str, Any],
    device: torch.device,
    regression_task: bool,
    classification_task: bool,
    loss_weighting: Literal["per_target", "per_function"] = "per_target",
    target_normalization: Literal[
        "per_function_zscore", "per_function_clamped", "none"
    ] = "per_function_zscore",
    target_std_floor: float = 1e-2,
    min_train_target_std: float = 0.0,
) -> torch.Tensor:
    return _compute_loss_result(
        wrapped_model=wrapped_model,
        criterion=criterion,
        full_data=full_data,
        device=device,
        regression_task=regression_task,
        classification_task=classification_task,
        loss_weighting=loss_weighting,
        target_normalization=target_normalization,
        target_std_floor=target_std_floor,
        min_train_target_std=min_train_target_std,
    ).loss


@torch.no_grad()
def _evaluate_prior_loss(
    wrapped_model: NanoTabPFNModel | nn.DataParallel[NanoTabPFNModel],
    val_prior: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    regression_task: bool,
    classification_task: bool,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    loss_weighting: Literal["per_target", "per_function"] = "per_target",
    target_normalization: Literal[
        "per_function_zscore", "per_function_clamped", "none"
    ] = "per_function_zscore",
    target_std_floor: float = 1e-2,
    min_train_target_std: float = 0.0,
) -> float:
    wrapped_model.eval()
    _set_criterion_mode(criterion, training=False)
    loss_sum = 0.0
    count = 0
    initial_pointer = None
    if hasattr(val_prior, "pointer"):
        pointer_value = val_prior.pointer
        if isinstance(pointer_value, int):
            initial_pointer = pointer_value
    try:
        for full_data in val_prior:
            autocast_ctx = (
                torch.autocast(
                    device_type=device.type,
                    dtype=amp_dtype,
                )
                if amp_enabled
                else nullcontext()
            )
            with autocast_ctx:
                current_loss = _compute_loss_result(
                    wrapped_model=wrapped_model,
                    criterion=criterion,
                    full_data=full_data,
                    device=device,
                    regression_task=regression_task,
                    classification_task=classification_task,
                    loss_weighting=loss_weighting,
                    target_normalization=target_normalization,
                    target_std_floor=target_std_floor,
                    min_train_target_std=min_train_target_std,
                )
            if torch.isnan(current_loss.loss) or current_loss.weight <= 0:
                continue
            loss_sum += float(current_loss.loss.detach().cpu().item()) * int(
                current_loss.weight
            )
            count += int(current_loss.weight)
    finally:
        if initial_pointer is not None:
            val_prior.pointer = initial_pointer
    if count == 0:
        return float("nan")
    return loss_sum / count


def train(
    model: NanoTabPFNModel,
    prior: DataLoader,
    criterion: nn.Module,
    epochs: int,
    accumulate_gradients: int = 1,
    lr: float = 1e-4,
    device: torch.device | None = None,
    callbacks: Sequence[Callback] | None = None,
    ckpt: dict[str, torch.Tensor] | None = None,
    multi_gpu: bool = False,
    run_name: str = "nanoTFM",
    val_prior: DataLoader | None = None,
    early_stopping: dict[str, Any] | None = None,
    eval_every_epochs: int = 1,
    max_train_seconds: int | None = None,
    weight_decay: float = 0.0,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float16,
    loss_weighting: Literal["per_target", "per_function"] = "per_target",
    optimizer_name: Literal["adamw_schedulefree", "adamw"] = "adamw_schedulefree",
    regression_loss_name: Literal["bar", "mse", "huber"] = "bar",
    target_normalization: Literal[
        "per_function_zscore", "per_function_clamped", "none"
    ] = "per_function_zscore",
    target_std_floor: float = 1e-2,
    min_train_target_std: float = 0.0,
    debug_trace_path: str | None = None,
    debug_trace_first_n_batches: int = 0,
    debug_trace_every_n_batches: int = 0,
):
    """
    Trains our model on the given prior using the given
    criterion.

    Args:
        model: (NanoTabPFNModel) our PyTorch model
        prior: (DataLoader) torch-compatible dataloader
        criterion: our loss criterion
        epochs: (int) the number of epochs we train for,
            the number of steps that constitute an epoch
            are decided by the prior
        accumulate_gradients: (int) the number of gradients
            to accumulate before updating the weights
        device: (torch.device) the device we are using
        callbacks: A list of callback instances to execute
            at the end of each epoch. These can be used
            for logging, validation, or other custom
            actions.
        ckpt: A checkpoint dictionary containing the model
            and optimizer states, as well as the last
            completed epoch. If provided, training resumes
            from this checkpoint.

    Returns:
        (torch.Tensor) shape (num_rows, batch_size,
            num_features, embedding_size)
    """
    work_dir = "workdir/" + run_name
    os.makedirs(work_dir, exist_ok=True)
    wrapped_model: NanoTabPFNModel | nn.DataParallel[NanoTabPFNModel] = model
    if multi_gpu:
        wrapped_model = nn.DataParallel(model)
    if callbacks is None:
        callbacks = []
    if not device:
        device = get_default_device()
    if not isinstance(device, torch.device):
        device = torch.device(device)
    wrapped_model.to(device)
    optimizer = _build_optimizer(
        optimizer_name=optimizer_name,
        parameters=wrapped_model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    if ckpt and ckpt.get("optimizer_name", optimizer_name) != optimizer_name:
        raise ValueError(
            "Checkpoint optimizer_name does not match requested optimizer_name."
        )
    if ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    classification_task = isinstance(criterion, nn.CrossEntropyLoss)
    regression_task = not classification_task
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(
        device="cuda",
        enabled=amp_enabled and amp_dtype == torch.float16,
    )
    if min_train_target_std < 0.0:
        raise ValueError("min_train_target_std must be >= 0.")
    if target_std_floor <= 0.0:
        raise ValueError("target_std_floor must be > 0.")
    if debug_trace_first_n_batches < 0:
        raise ValueError("debug_trace_first_n_batches must be >= 0.")
    if debug_trace_every_n_batches < 0:
        raise ValueError("debug_trace_every_n_batches must be >= 0.")

    assert prior.num_steps % accumulate_gradients == 0, (
        "num_steps must be divisible by accumulate_gradients"
    )

    if eval_every_epochs < 1:
        raise ValueError("eval_every_epochs must be >= 1.")
    start_time = time.time()
    latest_total_loss = 0.0
    stop_reason = "completed"
    stopped_early = False
    best_metric = ckpt.get("best_metric", float("inf")) if ckpt else float("inf")
    best_epoch = ckpt.get("best_epoch", 0) if ckpt else 0
    patience_counter = 0
    if ckpt:
        saved_early_stopping_state = ckpt.get("early_stopping_state", {})
        patience_counter = int(saved_early_stopping_state.get("patience_counter", 0))
    es_cfg = dict(early_stopping or {})
    metric_name = str(es_cfg.get("metric", "val_loss"))
    min_delta = float(es_cfg.get("min_delta", 1e-4))
    patience = int(es_cfg.get("patience", 10))
    min_epochs_before_stop = int(es_cfg.get("min_epochs_before_stop", 1))
    if min_epochs_before_stop < 1:
        raise ValueError("min_epochs_before_stop must be >= 1.")
    early_stopping_enabled = bool(es_cfg)
    debug_trace_records: list[dict[str, Any]] = []
    global_batch_index = 0

    try:
        for epoch in range(ckpt["epoch"] + 1 if ckpt else 1, epochs + 1):
            epoch_start_time = time.time()
            wrapped_model.train()  # Turn on the train mode
            _set_optimizer_mode(optimizer, training=True)
            _set_criterion_mode(criterion, training=True)
            total_loss = 0.0
            total_weight = 0
            pbar = tqdm(
                enumerate(prior), total=len(prior), desc=f"Epoch {epoch}", leave=False
            )
            for i, full_data in pbar:
                global_batch_index += 1
                trace_this_batch = bool(debug_trace_path) and (
                    _should_record_debug_trace(
                        global_batch_index=global_batch_index,
                        first_n_batches=debug_trace_first_n_batches,
                        every_n_batches=debug_trace_every_n_batches,
                    )
                )
                x_train = full_data["x"]
                y_train = full_data["y"]
                if torch.isnan(x_train).any() or torch.isnan(y_train).any():
                    if trace_this_batch:
                        debug_trace_records.append(
                            {
                                "epoch": epoch,
                                "batch_index": i,
                                "global_batch_index": global_batch_index,
                                "batch_size": int(x_train.shape[0]),
                                "valid_supervised_targets": 0,
                                "filtered_low_std_functions": 0,
                                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                                "skipped": True,
                                "skip_reason": "nan_input",
                            }
                        )
                    continue
                autocast_ctx = (
                    torch.autocast(
                        device_type=device.type,
                        dtype=amp_dtype,
                    )
                    if amp_enabled
                    else nullcontext()
                )
                with autocast_ctx:
                    loss_result = _compute_loss_result(
                        wrapped_model=wrapped_model,
                        criterion=criterion,
                        full_data=full_data,
                        device=device,
                        regression_task=regression_task,
                        classification_task=classification_task,
                        loss_weighting=loss_weighting,
                        target_normalization=target_normalization,
                        target_std_floor=target_std_floor,
                        min_train_target_std=min_train_target_std,
                    )
                if torch.isnan(loss_result.loss) or loss_result.weight <= 0:
                    if trace_this_batch:
                        trace_record = {
                            "epoch": epoch,
                            "batch_index": i,
                            "global_batch_index": global_batch_index,
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "skipped": True,
                            "skip_reason": "invalid_loss_or_weight",
                        }
                        if loss_result.debug is not None:
                            trace_record.update(loss_result.debug)
                        debug_trace_records.append(trace_record)
                    continue
                decoder_linear2 = _get_decoder_linear2(wrapped_model)
                decoder_weight_before_step = (
                    decoder_linear2.weight.detach().clone()
                    if trace_this_batch and decoder_linear2 is not None
                    else None
                )
                decoder_weight_norm = (
                    float(decoder_linear2.weight.detach().norm().item())
                    if trace_this_batch and decoder_linear2 is not None
                    else None
                )
                decoder_bias_norm = (
                    float(decoder_linear2.bias.detach().norm().item())
                    if (
                        trace_this_batch
                        and decoder_linear2 is not None
                        and decoder_linear2.bias is not None
                    )
                    else None
                )
                loss = loss_result.loss
                loss = loss / accumulate_gradients
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                grad_norm_before_clip = (
                    _parameter_grad_norm(wrapped_model.parameters())
                    if trace_this_batch
                    else None
                )
                grad_norm_after_clip = None
                decoder_weight_update_norm = None
                total_loss += (
                    float(loss_result.loss.detach().cpu().item()) * loss_result.weight
                )
                total_weight += loss_result.weight
                running_loss = total_loss / total_weight if total_weight > 0 else 0.0
                pbar.set_postfix({"loss": f"{running_loss:.4f}"})

                if (i + 1) % accumulate_gradients == 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(wrapped_model.parameters(), 1.0)
                    if trace_this_batch:
                        grad_norm_after_clip = _parameter_grad_norm(
                            wrapped_model.parameters()
                        )
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    if (
                        trace_this_batch
                        and decoder_linear2 is not None
                        and decoder_weight_before_step is not None
                    ):
                        decoder_weight_update_norm = float(
                            (
                                decoder_linear2.weight.detach()
                                - decoder_weight_before_step
                            )
                            .norm()
                            .item()
                        )
                    optimizer.zero_grad()
                if trace_this_batch:
                    trace_record = {
                        "epoch": epoch,
                        "batch_index": i,
                        "global_batch_index": global_batch_index,
                        "learning_rate": float(optimizer.param_groups[0]["lr"]),
                        "skipped": False,
                        "grad_norm_before_clip": grad_norm_before_clip,
                        "grad_norm_after_clip": grad_norm_after_clip,
                        "decoder_linear2_weight_norm": decoder_weight_norm,
                        "decoder_linear2_bias_norm": decoder_bias_norm,
                        "decoder_linear2_weight_update_norm": (
                            decoder_weight_update_norm
                        ),
                    }
                    if loss_result.debug is not None:
                        trace_record.update(loss_result.debug)
                    debug_trace_records.append(trace_record)

            end_time = time.time()
            mean_loss = total_loss / total_weight if total_weight > 0 else float("nan")
            latest_total_loss = mean_loss
            wrapped_model.eval()
            _set_optimizer_mode(optimizer, training=False)
            _set_criterion_mode(criterion, training=False)
            metrics: dict[str, float] = {"train_loss": float(mean_loss)}
            evaluated_this_epoch = val_prior is not None and (
                epoch % eval_every_epochs == 0
            )
            if evaluated_this_epoch:
                val_loss = _evaluate_prior_loss(
                    wrapped_model=wrapped_model,
                    val_prior=val_prior,
                    criterion=criterion,
                    device=device,
                    regression_task=regression_task,
                    classification_task=classification_task,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    loss_weighting=loss_weighting,
                    target_normalization=target_normalization,
                    target_std_floor=target_std_floor,
                    min_train_target_std=min_train_target_std,
                )
                metrics["val_loss"] = float(val_loss)
                metrics["val_rmse"] = (
                    float(val_loss**0.5) if val_loss >= 0 else float("nan")
                )
            if metric_name in metrics:
                current_metric = metrics[metric_name]
            else:
                current_metric = float("nan")
            improved = (
                evaluated_this_epoch
                and not torch.isnan(torch.tensor(current_metric))
                and (current_metric < (best_metric - min_delta))
            )
            if improved:
                best_metric = float(current_metric)
                best_epoch = epoch
                patience_counter = 0
            elif evaluated_this_epoch:
                patience_counter += 1

            training_state = {
                "epoch": epoch,
                "architecture": {
                    "num_layers": int(model.num_layers),
                    "embedding_size": int(model.embedding_size),
                    "num_attention_heads": int(model.num_attention_heads),
                    "mlp_hidden_size": int(model.mlp_hidden_size),
                    "num_outputs": int(model.num_outputs),
                    "dropout": float(getattr(model, "dropout", 0.0)),
                    "feature_normalization": getattr(
                        model,
                        "feature_normalization",
                        "per_function_zscore",
                    ),
                    "debug_output_clamp": getattr(model, "debug_output_clamp", None),
                },
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "optimizer_name": optimizer_name,
                "best_metric": float(best_metric),
                "best_epoch": int(best_epoch),
                "metrics": metrics,
                "loss_weighting": loss_weighting,
                "regression_loss": regression_loss_name,
                "target_normalization": target_normalization,
                "target_std_floor": float(target_std_floor),
                "min_train_target_std": float(min_train_target_std),
                "early_stopping_state": {
                    "enabled": early_stopping_enabled,
                    "metric": metric_name,
                    "patience": patience,
                    "min_delta": min_delta,
                    "min_epochs_before_stop": min_epochs_before_stop,
                    "patience_counter": int(patience_counter),
                },
            }
            torch.save(training_state, work_dir + "/latest_checkpoint.pth")
            if improved:
                torch.save(training_state, work_dir + "/best_checkpoint.pth")

            val_part = f"| val_loss {metrics.get('val_loss', float('nan')):.5f} "
            if regression_task and regression_loss_name == "mse":
                train_rmse = float("nan")
                if mean_loss >= 0:
                    train_rmse = float(mean_loss**0.5)
                val_rmse = float(metrics.get("val_rmse", float("nan")))
                val_part = (
                    f"| train_rmse {train_rmse:.5f} "
                    f"| val_loss {metrics.get('val_loss', float('nan')):.5f} "
                    f"| val_rmse {val_rmse:.5f} "
                )

            print(
                f"Epoch {epoch:5d} | time {end_time - epoch_start_time:5.2f}s "
                f"| train_loss {metrics['train_loss']:.5f} "
                f"{val_part}"
                f"| best_{metric_name} {best_metric:.5f} "
                f"| patience {patience_counter}/{patience}",
                flush=True,
            )

            for callback in callbacks:
                if type(criterion) is FullSupportBarDistribution:
                    callback.on_epoch_end(
                        epoch,
                        end_time - epoch_start_time,
                        mean_loss,
                        model,
                        dist=criterion,
                        metrics=metrics,
                    )
                else:
                    callback.on_epoch_end(
                        epoch,
                        end_time - epoch_start_time,
                        mean_loss,
                        model,
                        metrics=metrics,
                    )
            elapsed = time.time() - start_time
            if max_train_seconds is not None and elapsed >= max_train_seconds:
                stop_reason = "time_limit"
                break
            if (
                early_stopping_enabled
                and evaluated_this_epoch
                and epoch >= min_epochs_before_stop
                and patience_counter >= patience
            ):
                stopped_early = True
                stop_reason = "early_stopping"
                break
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
    finally:
        for callback in callbacks:
            callback.close()
    if debug_trace_path is not None:
        _write_debug_trace(
            path=debug_trace_path,
            metadata={
                "optimizer_name": optimizer_name,
                "regression_loss": regression_loss_name,
                "loss_weighting": loss_weighting,
                "target_normalization": target_normalization,
                "target_std_floor": float(target_std_floor),
                "min_train_target_std": float(min_train_target_std),
                "epochs": int(epochs),
                "steps_per_epoch": int(len(prior)),
            },
            records=debug_trace_records,
        )

    print(
        f"Training finished | reason={stop_reason} | best_epoch={best_epoch} "
        f"| best_{metric_name}={best_metric:.5f}",
        flush=True,
    )
    return model, {
        "latest_total_loss": float(latest_total_loss),
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
        "stopped_early": bool(stopped_early),
        "stop_reason": stop_reason,
        "loss_weighting": loss_weighting,
        "optimizer_name": optimizer_name,
        "regression_loss": regression_loss_name,
        "target_normalization": target_normalization,
        "target_std_floor": float(target_std_floor),
        "min_train_target_std": float(min_train_target_std),
        "debug_trace_path": debug_trace_path,
    }
