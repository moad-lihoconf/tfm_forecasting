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


def test_suite_metrics_reports_context_mean_and_ridge_baselines() -> None:
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
    )

    assert "baselines" in metrics
    baselines = metrics["baselines"]
    assert set(baselines.keys()) == {"context_mean", "ridge"}
    assert baselines["ridge"]["loss"] < baselines["context_mean"]["loss"]
    assert baselines["ridge"]["loss"] < metrics["loss"]
    assert baselines["ridge"]["nrmse"] < 0.1
