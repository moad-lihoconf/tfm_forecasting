"""Pytest coverage for DynSCM config coercion and graph sampling invariants."""

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
    """Load dynscm modules without importing tfmplayground.priors __init__."""
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
    return config_mod.DynSCMConfig, graph_mod.sample_regime_graphs


def _order_positions(order: np.ndarray) -> np.ndarray:
    positions = np.empty(order.shape[0], dtype=np.int64)
    positions[order] = np.arange(order.shape[0], dtype=np.int64)
    return positions


@pytest.fixture(scope="module")
def dynscm_api():
    return _load_dynscm_api()


def test_config_from_dict_coerces_tuples_and_rng_is_deterministic(dynscm_api):
    dynscm_config, _ = dynscm_api

    cfg = dynscm_config.from_dict(
        {
            "forecast_horizons": [1, 3, 6],
            "explicit_lags": [0, 1, 2],
            "season_period_choices": [7, 30],
            "random_seed": 13,
        }
    )

    assert cfg.forecast_horizons == (1, 3, 6)
    assert cfg.explicit_lags == (0, 1, 2)
    assert cfg.season_period_choices == (7, 30)

    first_draw = cfg.make_rng().integers(0, 1000, size=8)
    second_draw = cfg.make_rng().integers(0, 1000, size=8)
    assert np.array_equal(first_draw, second_draw)


def test_legacy_shape_keys_are_rejected(dynscm_api):
    dynscm_config, _ = dynscm_api

    with pytest.raises(ValueError):
        dynscm_config.from_dict({"horizon_choices": [1, 3, 6]})


def test_graph_sampling_is_seed_deterministic(dynscm_api):
    dynscm_config, sample_regime_graphs = dynscm_api

    cfg = dynscm_config()
    sample_one = sample_regime_graphs(cfg, num_vars=8, seed=17)
    sample_two = sample_regime_graphs(cfg, num_vars=8, seed=17)

    assert np.array_equal(
        sample_one.regime_topological_orders, sample_two.regime_topological_orders
    )
    assert np.array_equal(
        sample_one.base_contemporaneous_adjacency,
        sample_two.base_contemporaneous_adjacency,
    )
    assert np.array_equal(
        sample_one.base_lagged_adjacency, sample_two.base_lagged_adjacency
    )
    assert np.array_equal(
        sample_one.regime_contemporaneous_adjacency,
        sample_two.regime_contemporaneous_adjacency,
    )
    assert np.array_equal(
        sample_one.regime_lagged_adjacency, sample_two.regime_lagged_adjacency
    )


def test_graph_invariants_respected(dynscm_api):
    dynscm_config, sample_regime_graphs = dynscm_api

    cfg = dynscm_config()
    sample = sample_regime_graphs(cfg, num_vars=10, seed=5)

    assert sample.regime_topological_orders.shape == (cfg.num_regimes, 10)
    assert sample.regime_contemporaneous_adjacency.shape == (cfg.num_regimes, 10, 10)
    assert sample.regime_lagged_adjacency.shape == (
        cfg.num_regimes,
        cfg.max_lag,
        10,
        10,
    )

    for regime_idx in range(cfg.num_regimes):
        order = sample.regime_topological_orders[regime_idx]
        positions = _order_positions(order)
        contemporaneous = sample.regime_contemporaneous_adjacency[regime_idx]

        for source, target in np.argwhere(contemporaneous):
            assert positions[source] < positions[target]

        assert (contemporaneous.sum(axis=0) <= cfg.max_contemp_parents).all()
        assert (
            sample.regime_lagged_adjacency[regime_idx].sum(axis=1)
            <= cfg.max_lagged_parents
        ).all()

    assert len(sample.regime_parent_sets) == cfg.num_regimes
    assert len(sample.regime_parent_sets[0]) == 10
    for target_parents in sample.regime_parent_sets[0]:
        for _, lag in target_parents:
            assert 0 <= lag <= cfg.max_lag


def test_no_contemporaneous_mode_produces_no_contemporaneous_edges(dynscm_api):
    dynscm_config, sample_regime_graphs = dynscm_api

    cfg = dynscm_config(use_contemp_edges=False, share_base_graph=False)
    sample = sample_regime_graphs(cfg, num_vars=9, seed=23)

    assert not sample.base_contemporaneous_adjacency.any()
    assert not sample.regime_contemporaneous_adjacency.any()
