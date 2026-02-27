import os
import time
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any

import schedulefree
import torch
from pfns.bar_distribution import FullSupportBarDistribution
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from tfmplayground.callbacks import Callback
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.utils import get_default_device


def _compute_loss(
    wrapped_model: NanoTabPFNModel | nn.DataParallel[NanoTabPFNModel],
    criterion: nn.CrossEntropyLoss | FullSupportBarDistribution,
    full_data: dict[str, Any],
    device: torch.device,
    regression_task: bool,
    classification_task: bool,
) -> torch.Tensor:
    single_eval_pos = full_data["single_eval_pos"]
    x = full_data["x"].to(device)
    y = full_data["y"].to(device)
    target_y = full_data["target_y"].to(device)
    target_mask_full = full_data.get("target_mask")
    if target_mask_full is not None:
        target_mask_full = target_mask_full.to(device=device, dtype=torch.bool)
        if target_mask_full.ndim != 2:
            raise ValueError("target_mask must have shape (batch_size, num_rows).")
        if target_mask_full.shape != y.shape:
            raise ValueError("target_mask shape must match y/target_y shape.")

    def _loss_for_fixed_eval_pos(
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        target_y_batch: torch.Tensor,
        eval_pos: int,
        target_mask_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        data = (x_batch, y_batch[:, :eval_pos])
        targets = target_y_batch[:, eval_pos:]
        mask = None if target_mask_batch is None else target_mask_batch[:, eval_pos:]

        if regression_task:
            y_mean = data[1].mean(dim=1, keepdim=True)
            y_std = data[1].std(dim=1, keepdim=True) + 1e-8
            data = (data[0], (data[1] - y_mean) / y_std)
            targets = (targets - y_mean) / y_std

        output = wrapped_model(data, single_eval_pos=eval_pos)
        if classification_task:
            targets = targets.reshape((-1,)).to(torch.long)
            output = output.view(-1, output.shape[-1])
            if mask is not None:
                mask_flat = mask.reshape(-1)
                if not torch.any(mask_flat):
                    return torch.tensor(float("nan"), device=device)
                targets = targets[mask_flat]
                output = output[mask_flat]

        losses = criterion(output, targets)
        if mask is None or classification_task:
            return losses.mean() if losses.ndim > 0 else losses
        if losses.ndim == 0:
            return losses
        masked_losses = losses[mask]
        if masked_losses.numel() == 0:
            return torch.tensor(float("nan"), device=device)
        return masked_losses.mean()

    if isinstance(single_eval_pos, int):
        return _loss_for_fixed_eval_pos(
            x,
            y,
            target_y,
            int(single_eval_pos),
            target_mask_full,
        )

    if torch.is_tensor(single_eval_pos):
        if single_eval_pos.ndim == 0:
            return _loss_for_fixed_eval_pos(
                x,
                y,
                target_y,
                int(single_eval_pos.item()),
                target_mask_full,
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
        total = 0
        for eval_pos in unique_sep.tolist():
            mask = sep == int(eval_pos)
            if not torch.any(mask):
                continue
            batch_count = int(mask.sum().item())
            batch_loss = _loss_for_fixed_eval_pos(
                x[mask],
                y[mask],
                target_y[mask],
                int(eval_pos),
                target_mask_full[mask] if target_mask_full is not None else None,
            )
            weighted_loss = weighted_loss + batch_loss * batch_count
            total += batch_count
        if total == 0:
            return torch.tensor(float("nan"), device=device)
        return weighted_loss / total

    raise TypeError("single_eval_pos must be int or torch.Tensor.")


@torch.no_grad()
def _evaluate_prior_loss(
    wrapped_model: NanoTabPFNModel | nn.DataParallel[NanoTabPFNModel],
    val_prior: DataLoader,
    criterion: nn.CrossEntropyLoss | FullSupportBarDistribution,
    device: torch.device,
    regression_task: bool,
    classification_task: bool,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float16,
) -> float:
    wrapped_model.eval()
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
                current_loss = _compute_loss(
                    wrapped_model=wrapped_model,
                    criterion=criterion,
                    full_data=full_data,
                    device=device,
                    regression_task=regression_task,
                    classification_task=classification_task,
                )
            if torch.isnan(current_loss):
                continue
            loss_sum += float(current_loss.detach().cpu().item())
            count += 1
    finally:
        if initial_pointer is not None:
            val_prior.pointer = initial_pointer
    if count == 0:
        return float("nan")
    return loss_sum / count


def train(
    model: NanoTabPFNModel,
    prior: DataLoader,
    criterion: nn.CrossEntropyLoss | FullSupportBarDistribution,
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
    optimizer = schedulefree.AdamWScheduleFree(
        wrapped_model.parameters(), lr=lr, weight_decay=weight_decay
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

    try:
        for epoch in range(ckpt["epoch"] + 1 if ckpt else 1, epochs + 1):
            epoch_start_time = time.time()
            wrapped_model.train()  # Turn on the train mode
            optimizer.train()
            total_loss = 0.0
            pbar = tqdm(
                enumerate(prior), total=len(prior), desc=f"Epoch {epoch}", leave=False
            )
            for i, full_data in pbar:
                x_train = full_data["x"]
                y_train = full_data["y"]
                if torch.isnan(x_train).any() or torch.isnan(y_train).any():
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
                    loss = _compute_loss(
                        wrapped_model=wrapped_model,
                        criterion=criterion,
                        full_data=full_data,
                        device=device,
                        regression_task=regression_task,
                        classification_task=classification_task,
                    )
                loss = loss / accumulate_gradients
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                total_loss += loss.cpu().detach().item() * accumulate_gradients
                pbar.set_postfix({"loss": f"{total_loss / (i + 1):.4f}"})

                if (i + 1) % accumulate_gradients == 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(wrapped_model.parameters(), 1.0)
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

            end_time = time.time()
            mean_loss = total_loss / len(prior)
            latest_total_loss = total_loss
            wrapped_model.eval()
            optimizer.eval()
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
                },
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_metric": float(best_metric),
                "best_epoch": int(best_epoch),
                "metrics": metrics,
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

            print(
                f"Epoch {epoch:5d} | time {end_time - epoch_start_time:5.2f}s "
                f"| train_loss {metrics['train_loss']:.5f} "
                f"| val_loss {metrics.get('val_loss', float('nan')):.5f} "
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
    }
