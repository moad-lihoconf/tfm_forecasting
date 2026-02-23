"""Concise tests for DynSCM batch assembly."""

from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def dynscm_api(dynscm_modules):
    return dynscm_modules["config"], dynscm_modules["get_batch"]


def test_make_get_batch_dynscm_contract_shapes_and_padding(dynscm_api):
    config_mod, get_batch_mod = dynscm_api
    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "num_variables_min": 3,
            "num_variables_max": 3,
            "series_length_min": 80,
            "series_length_max": 80,
            "max_lag": 4,
            "mechanism_type": "linear_var",
            "missing_mode": "mix",
            "train_rows_min": 4,
            "train_rows_max": 4,
            "test_rows_min": 2,
            "test_rows_max": 2,
            "forecast_horizons": (1, 2, 3),
            "num_kernels": 1,
        }
    )
    get_batch = get_batch_mod.make_get_batch_dynscm(
        cfg,
        device=torch.device("cpu"),
        seed=7,
    )

    batch = get_batch(batch_size=3, num_datapoints_max=9, num_features=12)

    assert set(batch) == {"x", "y", "target_y", "single_eval_pos"}
    assert isinstance(batch["single_eval_pos"], int)

    x = batch["x"]
    y = batch["y"]
    target_y = batch["target_y"]

    assert x.shape == (3, 9, 12)
    assert y.shape == (3, 9)
    assert target_y.shape == (3, 9)
    assert batch["single_eval_pos"] == 4

    assert torch.equal(y, target_y)
    assert torch.isfinite(x).all()
    assert torch.isfinite(y).all()
    assert not torch.isnan(y[:, : batch["single_eval_pos"]]).any()

    # With fixed (n_train=4, n_test=2), tail rows [6:] must be zero-padded.
    assert torch.allclose(x[:, 6:, :], torch.zeros_like(x[:, 6:, :]))
    assert torch.allclose(y[:, 6:], torch.zeros_like(y[:, 6:]))


def test_make_get_batch_dynscm_is_deterministic_across_closures(dynscm_api):
    config_mod, get_batch_mod = dynscm_api
    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "num_variables_min": 3,
            "num_variables_max": 4,
            "series_length_min": 90,
            "series_length_max": 100,
            "max_lag": 5,
            "mechanism_type": "linear_var",
            "missing_mode": "mix",
            "train_rows_min": 6,
            "train_rows_max": 8,
            "test_rows_min": 3,
            "test_rows_max": 4,
        }
    )

    get_batch_a = get_batch_mod.make_get_batch_dynscm(
        cfg,
        device=torch.device("cpu"),
        seed=19,
    )
    get_batch_b = get_batch_mod.make_get_batch_dynscm(
        cfg,
        device=torch.device("cpu"),
        seed=19,
    )

    batch_a_1 = get_batch_a(batch_size=2, num_datapoints_max=20, num_features=24)
    batch_b_1 = get_batch_b(batch_size=2, num_datapoints_max=20, num_features=24)
    batch_a_2 = get_batch_a(batch_size=2, num_datapoints_max=20, num_features=24)
    batch_b_2 = get_batch_b(batch_size=2, num_datapoints_max=20, num_features=24)

    assert torch.equal(batch_a_1["x"], batch_b_1["x"])
    assert torch.equal(batch_a_1["y"], batch_b_1["y"])
    assert batch_a_1["single_eval_pos"] == batch_b_1["single_eval_pos"]

    assert torch.equal(batch_a_2["x"], batch_b_2["x"])
    assert torch.equal(batch_a_2["y"], batch_b_2["y"])
    assert batch_a_2["single_eval_pos"] == batch_b_2["single_eval_pos"]


def test_feature_priority_truncation_order_is_deterministic(dynscm_api):
    _, get_batch_mod = dynscm_api

    x = np.arange(1 * 2 * 10, dtype=np.float64).reshape(1, 2, 10)
    slices = {
        "data_values": (0, 6),
        "lags_value": (0, 4),
        "kern_value": (4, 6),
        "deterministic": (6, 8),
        "data_mask": (8, 10),
    }
    prioritized = get_batch_mod._prioritize_feature_blocks(
        x,
        feature_slices=slices,
    )

    expected = np.concatenate(
        [
            x[:, :, 6:8],  # deterministic first
            x[:, :, 0:4],  # explicit lags
            x[:, :, 4:6],  # kernels
            x[:, :, 8:10],  # masks
        ],
        axis=2,
    )
    assert np.array_equal(prioritized, expected)

    truncated = get_batch_mod._fit_feature_budget(prioritized, num_features=5)
    assert truncated.shape == (1, 2, 5)
    assert np.array_equal(truncated, expected[:, :, :5])


def test_get_batch_validates_inputs(dynscm_api):
    config_mod, get_batch_mod = dynscm_api
    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "num_variables_min": 3,
            "num_variables_max": 3,
            "series_length_min": 80,
            "series_length_max": 80,
            "max_lag": 4,
            "mechanism_type": "linear_var",
            "train_rows_min": 4,
            "train_rows_max": 4,
            "test_rows_min": 2,
            "test_rows_max": 2,
        }
    )
    get_batch = get_batch_mod.make_get_batch_dynscm(
        cfg, device=torch.device("cpu"), seed=99
    )

    with pytest.raises(ValueError, match="batch_size"):
        get_batch(batch_size=0, num_datapoints_max=10, num_features=8)

    with pytest.raises(ValueError, match="num_datapoints_max"):
        get_batch(batch_size=1, num_datapoints_max=1, num_features=8)

    with pytest.raises(ValueError, match="num_features"):
        get_batch(batch_size=1, num_datapoints_max=10, num_features=0)

    # Use valid config but num_datapoints_max too small for any feasible pair.
    with pytest.raises(ValueError, match="feasible"):
        get_batch(batch_size=1, num_datapoints_max=2, num_features=8)


@pytest.mark.parametrize(
    (
        "missing_mode",
        "use_contemp_edges",
        "drift_std",
        "num_kernels",
        "add_mask_channels",
    ),
    [
        ("off", True, 0.0, 0, False),
        ("mcar", False, 0.03, 2, True),
        ("mar", True, 0.0, 2, False),
        ("mnar_lite", False, 0.02, 0, True),
        ("mix", True, 0.05, 2, True),
    ],
)
def test_get_batch_scenario_matrix(
    dynscm_api,
    missing_mode: str,
    use_contemp_edges: bool,
    drift_std: float,
    num_kernels: int,
    add_mask_channels: bool,
):
    config_mod, get_batch_mod = dynscm_api
    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "num_variables_min": 3,
            "num_variables_max": 4,
            "series_length_min": 96,
            "series_length_max": 96,
            "max_lag": 6,
            "mechanism_type": "linear_var",
            "use_contemp_edges": use_contemp_edges,
            "drift_std": drift_std,
            "missing_mode": missing_mode,
            "num_kernels": num_kernels,
            "add_mask_channels": add_mask_channels,
            "train_rows_min": 8,
            "train_rows_max": 8,
            "test_rows_min": 4,
            "test_rows_max": 4,
        }
    )
    get_batch = get_batch_mod.make_get_batch_dynscm(
        cfg,
        device=torch.device("cpu"),
        seed=123,
    )

    batch = get_batch(batch_size=2, num_datapoints_max=16, num_features=24)
    x = batch["x"]
    y = batch["y"]
    split = int(batch["single_eval_pos"])

    assert x.shape == (2, 16, 24)
    assert y.shape == (2, 16)
    assert 0 < split < 16
    assert torch.isfinite(x).all()
    assert torch.isfinite(y).all()
    assert not torch.isnan(x).any()
    assert not torch.isnan(y[:, :split]).any()
