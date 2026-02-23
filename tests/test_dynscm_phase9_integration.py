"""Concise integration tests for DynSCM repository wiring."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


def _load_module(fullname: str, filepath: Path):
    spec = importlib.util.spec_from_file_location(fullname, filepath)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not create module spec for {fullname} from {filepath}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    spec.loader.exec_module(module)
    return module


def _load_priors_api():
    repo_root = Path(__file__).resolve().parents[1]
    priors_dir = repo_root / "tfmplayground" / "priors"
    dyn_dir = priors_dir / "dynscm"

    for pkg_name, pkg_path in (
        ("tfmplayground", repo_root / "tfmplayground"),
        ("tfmplayground.priors", priors_dir),
        ("tfmplayground.priors.dynscm", dyn_dir),
    ):
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(pkg_path)]
        sys.modules[pkg_name] = pkg

    # Load dynscm modules first so dataloader/main can resolve imports.
    config_mod = _load_module(
        "tfmplayground.priors.dynscm.config", dyn_dir / "config.py"
    )
    _load_module("tfmplayground.priors.dynscm.graph", dyn_dir / "graph.py")
    _load_module("tfmplayground.priors.dynscm.stability", dyn_dir / "stability.py")
    _load_module("tfmplayground.priors.dynscm.mechanisms", dyn_dir / "mechanisms.py")
    _load_module("tfmplayground.priors.dynscm.simulate", dyn_dir / "simulate.py")
    _load_module("tfmplayground.priors.dynscm.missingness", dyn_dir / "missingness.py")
    _load_module("tfmplayground.priors.dynscm.features", dyn_dir / "features.py")
    _load_module("tfmplayground.priors.dynscm.get_batch", dyn_dir / "get_batch.py")
    _load_module("tfmplayground.priors.dynscm", dyn_dir / "__init__.py")

    dataloader_mod = _load_module(
        "tfmplayground.priors.dataloader", priors_dir / "dataloader.py"
    )

    # Stub utils to avoid importing optional heavy deps (wandb/xgboost/ticl extras)
    # while testing DynSCM repository wiring.
    utils_stub = types.ModuleType("tfmplayground.priors.utils")
    utils_stub.build_tabpfn_prior = lambda *args, **kwargs: {}
    utils_stub.build_ticl_prior = lambda *args, **kwargs: {}
    utils_stub.dump_prior_to_h5 = lambda *args, **kwargs: None
    sys.modules["tfmplayground.priors.utils"] = utils_stub

    main_mod = _load_module("tfmplayground.priors.main", priors_dir / "main.py")
    root_mod = _load_module("tfmplayground.priors", priors_dir / "__init__.py")
    return config_mod, dataloader_mod, main_mod, root_mod


def test_dynscm_override_parser_and_config_loader():
    config_mod, _, main_mod, _ = _load_priors_api()

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


def test_dynscm_prior_dataloader_contract_and_root_exports():
    config_mod, dataloader_mod, _, root_mod = _load_priors_api()

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


def test_dynscm_override_parser_validates_inputs():
    _, _, main_mod, _ = _load_priors_api()

    with pytest.raises(ValueError, match="key=value"):
        main_mod._parse_dynscm_overrides(["no_equals_sign"])

    with pytest.raises(ValueError, match="empty"):
        main_mod._parse_dynscm_overrides(["=some_value"])
