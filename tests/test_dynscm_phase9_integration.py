"""Concise integration tests for DynSCM repository wiring."""

from __future__ import annotations

import pytest
import torch


def test_dynscm_override_parser_and_config_loader(priors_modules):
    config_mod = priors_modules["config"]
    main_mod = priors_modules["main"]

    parsed = main_mod._parse_dynscm_overrides(
        [
            "num_variables_min=5",
            "features.num_kernels=2",
            "add_time_feature=false",
            "forecast_horizons=[1,3,5]",
            'missing_mode="mar"',
        ]
    )

    assert parsed["num_variables_min"] == 5
    assert parsed["num_kernels"] == 2
    assert parsed["add_time_feature"] is False
    assert parsed["forecast_horizons"] == [1, 3, 5]
    assert parsed["missing_mode"] == "mar"

    cfg = main_mod._load_dynscm_config(None, ["num_variables_min=6", "num_kernels=1"])
    assert isinstance(cfg, config_mod.DynSCMConfig)
    assert cfg.num_variables_min == 6
    assert cfg.num_kernels == 1


def test_dynscm_prior_dataloader_contract_and_root_exports(priors_modules):
    config_mod = priors_modules["config"]
    dataloader_mod = priors_modules["dataloader"]
    root_mod = priors_modules["root"]

    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "num_variables_min": 3,
            "num_variables_max": 4,
            "series_length_min": 90,
            "series_length_max": 90,
            "max_lag": 5,
            "mechanism_type": "linear_var",
            "train_rows_min": 6,
            "train_rows_max": 6,
            "test_rows_min": 3,
            "test_rows_max": 3,
            "missing_mode": "mix",
        }
    )
    prior = dataloader_mod.DynSCMPriorDataLoader(
        cfg=cfg,
        num_steps=1,
        batch_size=2,
        num_datapoints_max=12,
        num_features=16,
        device=torch.device("cpu"),
        seed=9,
    )

    batch = next(iter(prior))
    assert set(batch) == {"x", "y", "target_y", "single_eval_pos"}
    assert batch["x"].shape == (2, 12, 16)
    assert batch["y"].shape == (2, 12)
    assert batch["target_y"].shape == (2, 12)
    assert isinstance(batch["single_eval_pos"], int)
    assert 0 < batch["single_eval_pos"] < 12
    assert torch.isfinite(batch["x"]).all()
    assert torch.isfinite(batch["y"]).all()
    assert torch.equal(batch["y"], batch["target_y"])

    assert hasattr(root_mod, "DynSCMPriorDataLoader")
    assert hasattr(root_mod, "DynSCMConfig")
    assert hasattr(root_mod, "make_get_batch_dynscm")


def test_dynscm_override_parser_validates_inputs(priors_modules):
    main_mod = priors_modules["main"]

    with pytest.raises(ValueError, match="key=value"):
        main_mod._parse_dynscm_overrides(["no_equals_sign"])

    with pytest.raises(ValueError, match="empty"):
        main_mod._parse_dynscm_overrides(["=some_value"])
