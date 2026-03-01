from __future__ import annotations

import json
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


class _ConstantClassificationModel(nn.Module):
    def forward(self, data, single_eval_pos: int):
        x, _y = data
        return torch.zeros(
            (x.shape[0], x.shape[1] - int(single_eval_pos), 1),
            dtype=torch.float32,
            device=x.device,
        )


class _SimpleRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
        self.num_layers = 1
        self.embedding_size = 8
        self.num_attention_heads = 1
        self.mlp_hidden_size = 16
        self.num_outputs = 1
        self.dropout = 0.0

    def forward(self, data, single_eval_pos: int):
        x, _y = data
        return self.linear(x).squeeze(-1)[:, int(single_eval_pos) :]


class _ClassificationTargetCriterion:
    def __call__(self, output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        del output
        return targets.to(torch.float32)


class _RecordingCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.mode_history: list[bool] = []

    def forward(self, output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        del targets
        self.mode_history.append(self.training)
        return torch.abs(output)


class _DtypeRecordingMSELoss(nn.MSELoss):
    def __init__(self):
        super().__init__(reduction="none")
        self.output_dtype: torch.dtype | None = None
        self.target_dtype: torch.dtype | None = None

    def forward(self, output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        self.output_dtype = output.dtype
        self.target_dtype = targets.dtype
        return super().forward(output, targets)


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


def test_compute_loss_supports_per_target_weighting_for_mixed_eval_positions():
    model = _ConstantClassificationModel()
    full_data = {
        "x": torch.randn(2, 4, 3, dtype=torch.float32),
        "y": torch.zeros((2, 4), dtype=torch.float32),
        "target_y": torch.tensor(
            [
                [0.0, 0.0, 10.0, 20.0],
                [0.0, 30.0, 40.0, 50.0],
            ],
            dtype=torch.float32,
        ),
        "single_eval_pos": torch.tensor([2, 1], dtype=torch.long),
    }

    per_function = train_mod._compute_loss(
        wrapped_model=model,
        criterion=_ClassificationTargetCriterion(),
        full_data=full_data,
        device=torch.device("cpu"),
        regression_task=False,
        classification_task=True,
        loss_weighting="per_function",
    )
    per_target = train_mod._compute_loss(
        wrapped_model=model,
        criterion=_ClassificationTargetCriterion(),
        full_data=full_data,
        device=torch.device("cpu"),
        regression_task=False,
        classification_task=True,
        loss_weighting="per_target",
    )

    assert float(per_function) == 27.5
    assert float(per_target) == 30.0


def test_compute_loss_matches_for_equal_target_counts():
    model = _ConstantClassificationModel()
    full_data = {
        "x": torch.randn(2, 4, 3, dtype=torch.float32),
        "y": torch.zeros((2, 4), dtype=torch.float32),
        "target_y": torch.tensor(
            [
                [0.0, 0.0, 10.0, 20.0],
                [0.0, 0.0, 30.0, 40.0],
            ],
            dtype=torch.float32,
        ),
        "single_eval_pos": 2,
    }

    per_function = train_mod._compute_loss(
        wrapped_model=model,
        criterion=_ClassificationTargetCriterion(),
        full_data=full_data,
        device=torch.device("cpu"),
        regression_task=False,
        classification_task=True,
        loss_weighting="per_function",
    )
    per_target = train_mod._compute_loss(
        wrapped_model=model,
        criterion=_ClassificationTargetCriterion(),
        full_data=full_data,
        device=torch.device("cpu"),
        regression_task=False,
        classification_task=True,
        loss_weighting="per_target",
    )

    assert float(per_function) == float(per_target)


def test_train_toggles_criterion_mode_between_train_and_eval():
    criterion = _RecordingCriterion()
    _, info = train_mod.train(
        model=_SimpleRegressionModel(),
        prior=_FixedPrior(num_steps=1, batch_size=2),
        val_prior=_FixedPrior(num_steps=1, batch_size=2),
        criterion=criterion,
        epochs=1,
        lr=1e-3,
        run_name="test_train_toggles_criterion_mode_between_train_and_eval",
        early_stopping={"metric": "val_loss", "patience": 5, "min_delta": 1e-4},
    )

    assert info["stop_reason"] == "completed"
    assert True in criterion.mode_history
    assert False in criterion.mode_history


def test_train_supports_plain_adamw_optimizer():
    _, info = train_mod.train(
        model=_SimpleRegressionModel(),
        prior=_FixedPrior(num_steps=1, batch_size=2),
        val_prior=_FixedPrior(num_steps=1, batch_size=2),
        criterion=_RecordingCriterion(),
        epochs=1,
        lr=1e-3,
        run_name="test_train_supports_plain_adamw_optimizer",
        optimizer_name="adamw",
        early_stopping={"metric": "val_loss", "patience": 5, "min_delta": 1e-4},
    )

    latest_ckpt = torch.load(
        Path("workdir")
        / "test_train_supports_plain_adamw_optimizer"
        / "latest_checkpoint.pth"
    )
    assert info["optimizer_name"] == "adamw"
    assert latest_ckpt["optimizer_name"] == "adamw"


def test_train_rejects_resume_optimizer_mismatch():
    with torch.no_grad():
        model = _SimpleRegressionModel()
    ckpt = {
        "epoch": 1,
        "model": model.state_dict(),
        "optimizer": torch.optim.AdamW(model.parameters(), lr=1e-3).state_dict(),
        "optimizer_name": "adamw_schedulefree",
    }

    try:
        train_mod.train(
            model=_SimpleRegressionModel(),
            prior=_FixedPrior(num_steps=1, batch_size=2),
            val_prior=_FixedPrior(num_steps=1, batch_size=2),
            criterion=_RecordingCriterion(),
            epochs=2,
            lr=1e-3,
            run_name="test_train_rejects_resume_optimizer_mismatch",
            optimizer_name="adamw",
            ckpt=ckpt,
            early_stopping={"metric": "val_loss", "patience": 5, "min_delta": 1e-4},
        )
    except ValueError as exc:
        assert "optimizer_name" in str(exc)
    else:
        raise AssertionError("Expected resume optimizer mismatch to raise ValueError")


def test_train_supports_scalar_regression_loss_metadata():
    _, info = train_mod.train(
        model=_SimpleRegressionModel(),
        prior=_FixedPrior(num_steps=1, batch_size=2),
        val_prior=_FixedPrior(num_steps=1, batch_size=2),
        criterion=nn.MSELoss(reduction="none"),
        epochs=1,
        lr=1e-3,
        run_name="test_train_supports_scalar_regression_loss_metadata",
        optimizer_name="adamw",
        regression_loss_name="mse",
        early_stopping={"metric": "val_loss", "patience": 5, "min_delta": 1e-4},
    )

    latest_ckpt = torch.load(
        Path("workdir")
        / "test_train_supports_scalar_regression_loss_metadata"
        / "latest_checkpoint.pth"
    )
    assert info["regression_loss"] == "mse"
    assert latest_ckpt["regression_loss"] == "mse"


def test_compute_loss_casts_scalar_regression_to_float32():
    model = _SimpleRegressionModel()
    criterion = _DtypeRecordingMSELoss()
    full_data = {
        "x": torch.randn(2, 5, 3, dtype=torch.float32),
        "y": torch.randn(2, 5, dtype=torch.float32),
        "target_y": torch.randn(2, 5, dtype=torch.float32),
        "single_eval_pos": 3,
    }

    class _HalfOutputModel(nn.Module):
        def forward(self, data, single_eval_pos: int):
            out = model(data, single_eval_pos)
            return out.to(torch.float16)

    loss = train_mod._compute_loss(
        wrapped_model=_HalfOutputModel(),
        criterion=criterion,
        full_data=full_data,
        device=torch.device("cpu"),
        regression_task=True,
        classification_task=False,
        loss_weighting="per_target",
    )

    assert torch.isfinite(loss)
    assert criterion.output_dtype == torch.float32
    assert criterion.target_dtype == torch.float32


def test_compute_loss_skips_low_std_regression_functions():
    model = _SimpleRegressionModel()
    full_data = {
        "x": torch.randn(2, 4, 3, dtype=torch.float32),
        "y": torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 2.0, 3.0],
            ],
            dtype=torch.float32,
        ),
        "target_y": torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 2.0, 3.0],
            ],
            dtype=torch.float32,
        ),
        "single_eval_pos": 2,
    }

    unfiltered = train_mod._compute_loss_result(
        wrapped_model=model,
        criterion=nn.MSELoss(reduction="none"),
        full_data=full_data,
        device=torch.device("cpu"),
        regression_task=True,
        classification_task=False,
        loss_weighting="per_target",
    )
    filtered = train_mod._compute_loss_result(
        wrapped_model=model,
        criterion=nn.MSELoss(reduction="none"),
        full_data=full_data,
        device=torch.device("cpu"),
        regression_task=True,
        classification_task=False,
        loss_weighting="per_target",
        min_train_target_std=0.1,
    )

    assert torch.isfinite(unfiltered.loss)
    assert torch.isfinite(filtered.loss)
    assert unfiltered.weight == 4
    assert filtered.weight == 2
    assert filtered.debug["filtered_low_std_functions"] == 1


def test_compute_loss_supports_target_normalization_modes():
    model = _SimpleRegressionModel()
    full_data = {
        "x": torch.randn(1, 4, 3, dtype=torch.float32),
        "y": torch.tensor(
            [
                [1.0, 1.0001, 1.0002, 1.0003],
            ],
            dtype=torch.float32,
        ),
        "target_y": torch.tensor(
            [
                [1.0, 1.0001, 1.0002, 1.0003],
            ],
            dtype=torch.float32,
        ),
        "single_eval_pos": 2,
    }

    zscore = train_mod._compute_loss_result(
        wrapped_model=model,
        criterion=nn.MSELoss(reduction="none"),
        full_data=full_data,
        device=torch.device("cpu"),
        regression_task=True,
        classification_task=False,
        target_normalization="per_function_zscore",
    )
    clamped = train_mod._compute_loss_result(
        wrapped_model=model,
        criterion=nn.MSELoss(reduction="none"),
        full_data=full_data,
        device=torch.device("cpu"),
        regression_task=True,
        classification_task=False,
        target_normalization="per_function_clamped",
        target_std_floor=1e-2,
    )
    unnormalized = train_mod._compute_loss_result(
        wrapped_model=model,
        criterion=nn.MSELoss(reduction="none"),
        full_data=full_data,
        device=torch.device("cpu"),
        regression_task=True,
        classification_task=False,
        target_normalization="none",
    )

    assert torch.isfinite(zscore.loss)
    assert torch.isfinite(clamped.loss)
    assert torch.isfinite(unnormalized.loss)
    assert zscore.debug is not None
    assert clamped.debug is not None
    assert unnormalized.debug is not None
    assert (
        zscore.debug["normalized_target_abs"]["max"]
        > clamped.debug["normalized_target_abs"]["max"]
    )


def test_train_writes_debug_trace_json(tmp_path: Path):
    trace_path = tmp_path / "trace.json"
    _, info = train_mod.train(
        model=_SimpleRegressionModel(),
        prior=_FixedPrior(num_steps=1, batch_size=2),
        val_prior=_FixedPrior(num_steps=1, batch_size=2),
        criterion=nn.MSELoss(reduction="none"),
        epochs=1,
        lr=1e-3,
        run_name="test_train_writes_debug_trace_json",
        optimizer_name="adamw",
        regression_loss_name="mse",
        debug_trace_path=str(trace_path),
        debug_trace_first_n_batches=1,
        target_normalization="none",
        early_stopping={"metric": "val_loss", "patience": 5, "min_delta": 1e-4},
    )

    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert info["debug_trace_path"] == str(trace_path)
    assert payload["metadata"]["target_normalization"] == "none"
    assert len(payload["records"]) == 1
    record = payload["records"][0]
    assert record["epoch"] == 1
    assert record["batch_index"] == 0
    assert record["global_batch_index"] == 1
    assert "train_target_std" in record
    assert "normalized_target_abs" in record
    assert "output_abs" in record
    assert "grad_norm_before_clip" in record
    assert "decoder_linear2_weight_update_norm" in record


def test_feature_encoder_supports_none_normalization():
    model = NanoTabPFNModel(
        num_attention_heads=1,
        embedding_size=4,
        mlp_hidden_size=8,
        num_layers=1,
        num_outputs=1,
        feature_normalization="none",
    )
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
    encoded = model.feature_encoder(x, single_eval_pos=1)
    expected = model.feature_encoder.linear_layer(x.unsqueeze(-1))
    assert torch.allclose(encoded, expected)


def test_decoder_output_clamp_applies_to_scalar_outputs():
    model = NanoTabPFNModel(
        num_attention_heads=1,
        embedding_size=4,
        mlp_hidden_size=8,
        num_layers=1,
        num_outputs=1,
        debug_output_clamp=0.1,
    )
    decoder_input = torch.full((2, 3, 4), 100.0, dtype=torch.float32)
    output = model.decoder(decoder_input)
    assert float(output.abs().max().item()) <= 0.100001
