"""Concise tests for DynSCM phase-3 stability constraints."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


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


def _load_dynscm_api():
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

    config_mod = _load_module(
        "tfmplayground.priors.dynscm.config", dyn_dir / "config.py"
    )
    graph_mod = _load_module("tfmplayground.priors.dynscm.graph", dyn_dir / "graph.py")
    stability_mod = _load_module(
        "tfmplayground.priors.dynscm.stability", dyn_dir / "stability.py"
    )
    return config_mod, graph_mod, stability_mod


@pytest.fixture(scope="module")
def dynscm_api():
    return _load_dynscm_api()


def test_sample_stable_coefficients_is_seed_deterministic(dynscm_api):
    config_mod, graph_mod, stability_mod = dynscm_api
    cfg = config_mod.DynSCMConfig()

    graph = graph_mod.sample_regime_graphs(cfg, num_vars=8, seed=11)
    s1 = stability_mod.sample_stable_coefficients(cfg, graph, seed=19)
    s2 = stability_mod.sample_stable_coefficients(cfg, graph, seed=19)

    assert np.array_equal(s1.lag_coeffs, s2.lag_coeffs)
    assert np.array_equal(s1.contemp_coeffs, s2.contemp_coeffs)
    assert np.array_equal(s1.lag_column_budgets, s2.lag_column_budgets)
    assert np.array_equal(s1.contemp_column_budgets, s2.contemp_column_budgets)


def test_column_budget_support_and_budget_invariants(dynscm_api):
    config_mod, graph_mod, stability_mod = dynscm_api
    cfg = config_mod.DynSCMConfig()
    graph = graph_mod.sample_regime_graphs(cfg, num_vars=10, seed=3)
    sample = stability_mod.sample_stable_coefficients(cfg, graph, seed=7)

    assert sample.lag_coeffs.shape == graph.regime_lagged_adjacency.shape
    assert sample.contemp_coeffs.shape == (
        cfg.num_regimes,
        10,
        10,
    )

    tol = 1e-8
    assert not (
        (~graph.regime_lagged_adjacency) & (np.abs(sample.lag_coeffs) > tol)
    ).any()
    assert not (
        (~graph.regime_contemp_adjacency) & (np.abs(sample.contemp_coeffs) > tol)
    ).any()

    lag_l1 = np.sum(np.abs(sample.lag_coeffs), axis=(1, 2))
    cont_l1 = np.sum(np.abs(sample.contemp_coeffs), axis=1)
    assert np.all(lag_l1 <= sample.lag_column_budgets + tol)
    assert np.all(cont_l1 <= sample.contemp_column_budgets + tol)


def test_spectral_rescale_lag_block_reduces_radius(dynscm_api):
    _, _, stability_mod = dynscm_api

    lag = np.zeros((2, 2, 2), dtype=np.float64)
    lag[0, 0, 0] = 1.6
    lag[0, 1, 1] = 1.4

    rho_before = stability_mod.companion_spectral_radius(lag)
    lag_rescaled, scale, rho_after = stability_mod.rescale_lag_block_to_spectral_radius(
        lag, spectral_radius_cap=0.95
    )

    assert rho_before > 0.95
    assert scale < 1.0
    assert rho_after <= 0.95 + 1e-6
    assert np.max(np.abs(lag_rescaled)) <= np.max(np.abs(lag))


def test_spectral_rescale_enabled_in_sampling_enforces_cap(dynscm_api):
    config_mod, graph_mod, stability_mod = dynscm_api
    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "enable_spectral_rescale": True,
            "spectral_radius_cap": 0.90,
            "col_budget_min": 0.90,
            "col_budget_max": 0.99,
        }
    )
    graph = graph_mod.sample_regime_graphs(cfg, num_vars=9, seed=13)
    sample = stability_mod.sample_stable_coefficients(cfg, graph, seed=14)
    assert np.all(sample.lag_spectral_radii <= cfg.spectral_radius_cap + 1e-6)
