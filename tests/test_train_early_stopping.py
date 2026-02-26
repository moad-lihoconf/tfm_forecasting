from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import tfmplayground.train as train_mod
from tfmplayground.model import NanoTabPFNModel


class _FixedPrior:
    def __init__(self, *, num_steps: int, batch_size: int = 2):
        self.num_steps = num_steps
        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(self.num_steps):
            x = torch.randn(self.batch_size, 5, 3, dtype=torch.float32)
            y = torch.randint(0, 3, (self.batch_size, 5), dtype=torch.float32)
            yield {
                "x": x,
                "y": y,
                "target_y": y.clone(),
                "single_eval_pos": 3,
            }

    def __len__(self):
        return self.num_steps


def _model() -> NanoTabPFNModel:
    return NanoTabPFNModel(
        num_attention_heads=1,
        embedding_size=8,
        mlp_hidden_size=16,
        num_layers=1,
        num_outputs=3,
        dropout=0.1,
    )


def test_early_stops_after_patience(monkeypatch):
    values = iter([1.0, 1.0, 1.0, 1.0, 1.0])
    monkeypatch.setattr(
        train_mod, "_evaluate_prior_loss", lambda **kwargs: next(values)
    )

    run_name = "test_early_stop_after_patience"
    model, info = train_mod.train(
        model=_model(),
        prior=_FixedPrior(num_steps=1),
        val_prior=_FixedPrior(num_steps=1),
        criterion=nn.CrossEntropyLoss(),
        epochs=10,
        lr=1e-3,
        run_name=run_name,
        early_stopping={"metric": "val_loss", "patience": 2, "min_delta": 1e-4},
    )

    assert isinstance(model, NanoTabPFNModel)
    assert info["stopped_early"] is True
    assert info["stop_reason"] == "early_stopping"
    assert info["best_epoch"] == 1


def test_improvement_updates_best_checkpoint(monkeypatch):
    values = iter([1.0, 0.8, 0.85])
    monkeypatch.setattr(
        train_mod, "_evaluate_prior_loss", lambda **kwargs: next(values)
    )

    run_name = "test_improvement_updates_best_checkpoint"
    _, info = train_mod.train(
        model=_model(),
        prior=_FixedPrior(num_steps=1),
        val_prior=_FixedPrior(num_steps=1),
        criterion=nn.CrossEntropyLoss(),
        epochs=3,
        lr=1e-3,
        run_name=run_name,
        early_stopping={"metric": "val_loss", "patience": 10, "min_delta": 1e-4},
    )

    best_ckpt = torch.load(Path("workdir") / run_name / "best_checkpoint.pth")
    assert info["best_epoch"] == 2
    assert best_ckpt["epoch"] == 2
    assert best_ckpt["best_epoch"] == 2


def test_resume_restores_early_stopping_state(monkeypatch):
    values_a = iter([1.0, 1.0])
    monkeypatch.setattr(
        train_mod, "_evaluate_prior_loss", lambda **kwargs: next(values_a)
    )
    run_name = "test_resume_restores_early_stopping_state"
    _, first_info = train_mod.train(
        model=_model(),
        prior=_FixedPrior(num_steps=1),
        val_prior=_FixedPrior(num_steps=1),
        criterion=nn.CrossEntropyLoss(),
        epochs=2,
        lr=1e-3,
        run_name=run_name,
        early_stopping={"metric": "val_loss", "patience": 2, "min_delta": 1e-4},
    )
    assert first_info["stopped_early"] is False
    latest_ckpt = torch.load(Path("workdir") / run_name / "latest_checkpoint.pth")

    values_b = iter([1.0])
    monkeypatch.setattr(
        train_mod, "_evaluate_prior_loss", lambda **kwargs: next(values_b)
    )
    _, resumed_info = train_mod.train(
        model=_model(),
        prior=_FixedPrior(num_steps=1),
        val_prior=_FixedPrior(num_steps=1),
        criterion=nn.CrossEntropyLoss(),
        epochs=3,
        lr=1e-3,
        run_name=run_name,
        early_stopping={"metric": "val_loss", "patience": 2, "min_delta": 1e-4},
        ckpt=latest_ckpt,
    )
    assert resumed_info["stopped_early"] is True
    assert resumed_info["stop_reason"] == "early_stopping"


def test_honors_max_train_seconds(monkeypatch):
    values = iter([1.0, 1.0, 1.0])
    monkeypatch.setattr(
        train_mod, "_evaluate_prior_loss", lambda **kwargs: next(values)
    )
    _, info = train_mod.train(
        model=_model(),
        prior=_FixedPrior(num_steps=1),
        val_prior=_FixedPrior(num_steps=1),
        criterion=nn.CrossEntropyLoss(),
        epochs=10,
        lr=1e-3,
        run_name="test_max_train_seconds",
        early_stopping={"metric": "val_loss", "patience": 100, "min_delta": 1e-4},
        max_train_seconds=0,
    )
    assert info["stop_reason"] == "time_limit"


def test_compute_loss_supports_per_sample_single_eval_pos():
    model = _model().to(torch.device("cpu"))
    full_data = {
        "x": torch.randn(2, 5, 3, dtype=torch.float32),
        "y": torch.randint(0, 3, (2, 5), dtype=torch.float32),
        "target_y": torch.randint(0, 3, (2, 5), dtype=torch.float32),
        "single_eval_pos": torch.tensor([2, 3], dtype=torch.long),
    }
    loss = train_mod._compute_loss(
        wrapped_model=model,
        criterion=nn.CrossEntropyLoss(),
        full_data=full_data,
        device=torch.device("cpu"),
        regression_task=False,
        classification_task=True,
    )
    assert torch.isfinite(loss)
