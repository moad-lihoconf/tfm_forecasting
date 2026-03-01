"""Shared fixtures for the DynSCM test suite."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _load_module(fullname: str, filepath: Path):
    """Load a Python module by file path without requiring package installation."""
    spec = importlib.util.spec_from_file_location(fullname, filepath)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not create module spec for {fullname} from {filepath}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    spec.loader.exec_module(module)
    return module


def _register_package_stubs() -> Path:
    """Register package stubs in sys.modules so relative imports resolve."""
    repo_root = Path(__file__).resolve().parents[1]
    dyn_dir = repo_root / "tfmplayground" / "priors" / "dynscm"

    for pkg_name, pkg_path in (
        ("tfmplayground", repo_root / "tfmplayground"),
        ("tfmplayground.priors", repo_root / "tfmplayground" / "priors"),
        ("tfmplayground.priors.dynscm", dyn_dir),
    ):
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(pkg_path)]
        sys.modules[pkg_name] = pkg

    return repo_root


@pytest.fixture(scope="session")
def dynscm_modules():
    """Load all DynSCM submodules once per test session."""
    repo_root = _register_package_stubs()
    dyn_dir = repo_root / "tfmplayground" / "priors" / "dynscm"

    modules: dict[str, types.ModuleType] = {}
    for name in (
        "config",
        "graph",
        "stability",
        "mechanisms",
        "simulate",
        "missingness",
        "features",
        "research",
        "get_batch",
    ):
        modules[name] = _load_module(
            f"tfmplayground.priors.dynscm.{name}", dyn_dir / f"{name}.py"
        )

    _load_module("tfmplayground.priors.dynscm", dyn_dir / "__init__.py")
    return modules


@pytest.fixture(scope="session")
def priors_modules(dynscm_modules):
    """Load priors-level modules (dataloader, main, root) on top of DynSCM."""
    repo_root = Path(__file__).resolve().parents[1]
    priors_dir = repo_root / "tfmplayground" / "priors"

    dataloader_mod = _load_module(
        "tfmplayground.priors.dataloader", priors_dir / "dataloader.py"
    )

    # Stub utils to avoid importing optional heavy deps (wandb/xgboost/ticl).
    utils_stub = types.ModuleType("tfmplayground.priors.utils")
    utils_stub.build_tabpfn_prior = lambda *args, **kwargs: {}
    utils_stub.build_ticl_prior = lambda *args, **kwargs: {}
    utils_stub.dump_prior_to_h5 = lambda *args, **kwargs: None
    sys.modules["tfmplayground.priors.utils"] = utils_stub

    main_mod = _load_module("tfmplayground.priors.main", priors_dir / "main.py")
    research_profiles_mod = _load_module(
        "tfmplayground.priors.dynscm.research_profiles",
        priors_dir / "dynscm" / "research_profiles.py",
    )
    root_mod = _load_module("tfmplayground.priors", priors_dir / "__init__.py")

    return {
        **dynscm_modules,
        "dataloader": dataloader_mod,
        "main": main_mod,
        "research_profiles": research_profiles_mod,
        "root": root_mod,
    }


def order_positions(order: np.ndarray) -> np.ndarray:
    """Invert a permutation: positions[i] = index where i appears in order."""
    positions = np.empty(order.shape[0], dtype=np.int64)
    positions[order] = np.arange(order.shape[0], dtype=np.int64)
    return positions
