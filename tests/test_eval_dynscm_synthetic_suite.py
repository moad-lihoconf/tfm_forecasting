from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def _load_script_module(script_name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


eval_mod = _load_script_module("eval_dynscm_synthetic_suite.py")


class _ZeroModel(torch.nn.Module):
    def forward(self, data, single_eval_pos: int):  # type: ignore[override]
        x, y_context = data
        del x
        batch_size = y_context.shape[0]
        query_len = 4 - int(single_eval_pos)
        return torch.zeros((batch_size, query_len), dtype=torch.float32)


class _CaptureModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.captured_context: torch.Tensor | None = None

    def forward(self, data, single_eval_pos: int):  # type: ignore[override]
        x, y_context = data
        del x
        self.captured_context = y_context.detach().clone()
        batch_size = y_context.shape[0]
        query_len = 4 - int(single_eval_pos)
        return torch.zeros((batch_size, query_len), dtype=torch.float32)


def test_robust_ridge_baseline_does_not_blow_up_on_tiny_std_feature_shift() -> None:
    from tfmplayground.priors.dynscm.difficulty import ridge_holdout_predictions

    x = torch.tensor(
        [
            [[0.0, 1e-6], [1.0, 2e-6], [2.0, 3e-6], [100.0, 100.0]],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32)
    batch = {
        "x": x,
        "y": y,
        "target_y": y.clone(),
        "single_eval_pos": 3,
        "num_datapoints": 4,
        "target_mask": torch.tensor([[False, False, False, True]], dtype=torch.bool),
    }

    preds = ridge_holdout_predictions(
        x[0].numpy(),
        y[0].numpy(),
        n_train=3,
        alpha=1e-3,
        std_floor=1e-3,
        z_clip=5.0,
    )
    assert preds.shape == (1,)
    assert abs(float(preds[0])) < 20.0

    metrics = eval_mod._suite_metrics(
        model=_ZeroModel(),
        loader=[batch],
        device=torch.device("cpu"),
        target_normalization="none",
        target_std_floor=1e-2,
    )
    assert metrics["baselines"]["ridge"]["loss"] < 400.0


def test_suite_metrics_normalizes_context_and_unnormalizes_predictions() -> None:
    x = torch.tensor(
        [[[0.0], [1.0], [2.0], [3.0]]],
        dtype=torch.float32,
    )
    y = torch.tensor([[10.0, 20.0, 30.0, 40.0]], dtype=torch.float32)
    batch = {
        "x": x,
        "y": y,
        "target_y": y.clone(),
        "single_eval_pos": 2,
        "num_datapoints": 4,
        "target_mask": torch.tensor([[False, False, True, True]], dtype=torch.bool),
    }
    model = _CaptureModel()

    metrics = eval_mod._suite_metrics(
        model=model,
        loader=[batch],
        device=torch.device("cpu"),
        target_normalization="per_function_clamped",
        target_std_floor=5.0,
    )
    assert model.captured_context is not None
    expected_context = torch.tensor([[-0.70710677, 0.70710677]], dtype=torch.float32)
    assert torch.allclose(model.captured_context, expected_context, atol=1e-5)
    assert metrics["rmse"] > 20.0


def test_resolve_eval_target_settings_prefers_training_metadata() -> None:
    target_normalization, target_std_floor, source = (
        eval_mod._resolve_eval_target_settings(
            {
                "training": {
                    "target_normalization": "per_function_clamped",
                    "target_std_floor": 0.05,
                },
                "live_profile": {
                    "cli": {
                        "target_normalization": "none",
                        "target_std_floor": 0.01,
                    }
                },
            }
        )
    )
    assert target_normalization == "per_function_clamped"
    assert target_std_floor == 0.05
    assert source == "training"


def test_resolve_eval_target_settings_falls_back_to_live_profile_cli() -> None:
    target_normalization, target_std_floor, source = (
        eval_mod._resolve_eval_target_settings(
            {
                "training": {},
                "live_profile": {
                    "cli": {
                        "target_normalization": "per_function_zscore",
                        "target_std_floor": 0.02,
                    }
                },
            }
        )
    )
    assert target_normalization == "per_function_zscore"
    assert target_std_floor == 0.02
    assert source == "live_profile.cli"


def test_suite_metrics_reports_context_mean_and_ridge_baselines_with_none_normalization() -> (
    None
):
    x = torch.tensor(
        [
            [[0.0], [1.0], [2.0], [3.0]],
            [[10.0], [11.0], [12.0], [13.0]],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [10.0, 11.0, 12.0, 13.0],
        ],
        dtype=torch.float32,
    )
    batch = {
        "x": x,
        "y": y,
        "target_y": y.clone(),
        "single_eval_pos": 2,
        "num_datapoints": 4,
        "target_mask": torch.tensor(
            [[False, False, True, True], [False, False, True, True]],
            dtype=torch.bool,
        ),
    }

    metrics = eval_mod._suite_metrics(
        model=_ZeroModel(),
        loader=[batch],
        device=torch.device("cpu"),
        target_normalization="none",
        target_std_floor=1e-2,
    )
    assert "baselines" in metrics
    baselines = metrics["baselines"]
    assert set(baselines.keys()) == {"context_mean", "ridge"}
    assert baselines["ridge"]["loss"] < baselines["context_mean"]["loss"]
    assert baselines["ridge"]["loss"] < metrics["loss"]
    assert baselines["ridge"]["nrmse"] < 0.3
