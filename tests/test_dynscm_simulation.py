"""Concise tests for DynSCM phase-5 simulation engine."""

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
    mechanisms_mod = _load_module(
        "tfmplayground.priors.dynscm.mechanisms", dyn_dir / "mechanisms.py"
    )
    simulate_mod = _load_module(
        "tfmplayground.priors.dynscm.simulate", dyn_dir / "simulate.py"
    )
    return config_mod, graph_mod, stability_mod, mechanisms_mod, simulate_mod


@pytest.fixture(scope="module")
def dynscm_api():
    return _load_dynscm_api()


def test_sample_regime_path_respects_sticky_extremes(dynscm_api):
    config_mod, _, _, _, simulate_mod = dynscm_api

    sticky_cfg = config_mod.DynSCMConfig.from_dict(
        {"num_regimes": 3, "sticky_rho": 1.0}
    )
    sticky_path = simulate_mod.sample_regime_path(sticky_cfg, num_steps=48, seed=5)
    assert np.all(sticky_path == sticky_path[0])

    switch_cfg = config_mod.DynSCMConfig.from_dict(
        {"num_regimes": 4, "sticky_rho": 0.0}
    )
    switch_path = simulate_mod.sample_regime_path(switch_cfg, num_steps=64, seed=7)
    assert np.all(switch_path[1:] != switch_path[:-1])
    assert switch_path.min() >= 0
    assert switch_path.max() < switch_cfg.num_regimes


def test_simulate_dynscm_series_is_seed_deterministic_and_finite(dynscm_api):
    config_mod, graph_mod, _, mechanisms_mod, simulate_mod = dynscm_api

    cfg = config_mod.DynSCMConfig.from_dict({"mechanism_type": "linear_var"})
    graph = graph_mod.sample_regime_graphs(cfg, num_vars=6, seed=11)
    mechanism = mechanisms_mod.sample_regime_mechanisms(cfg, graph, seed=12)

    fixed_path = np.zeros((80,), dtype=np.int64)
    sim_one = simulate_mod.simulate_dynscm_series(
        cfg,
        graph,
        mechanism,
        num_steps=80,
        regime_path=fixed_path,
        seed=13,
    )
    sim_two = simulate_mod.simulate_dynscm_series(
        cfg,
        graph,
        mechanism,
        num_steps=80,
        regime_path=fixed_path,
        seed=13,
    )

    assert np.array_equal(sim_one.series, sim_two.series)
    assert np.array_equal(sim_one.regime_path, sim_two.regime_path)
    assert np.array_equal(sim_one.innovations, sim_two.innovations)
    assert np.array_equal(sim_one.noise_scales, sim_two.noise_scales)
    assert np.array_equal(
        sim_one.initial_lagged_history,
        sim_two.initial_lagged_history,
    )

    assert sim_one.series.shape == (80, 6)
    assert sim_one.regime_path.shape == (80,)
    assert not sim_one.clipped
    assert np.isfinite(sim_one.series).all()
    assert sim_one.max_abs_value <= cfg.max_abs_x


def test_simulation_clips_when_resampling_budget_is_exhausted(dynscm_api):
    config_mod, graph_mod, _, mechanisms_mod, simulate_mod = dynscm_api

    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "mechanism_type": "linear_var",
            "num_regimes": 2,
            "sticky_rho": 1.0,
            "noise_family": "normal",
            "noise_scale_min": 400.0,
            "noise_scale_max": 400.0,
            "max_abs_x": 0.1,
            "max_resample_attempts": 1,
        }
    )
    graph = graph_mod.sample_regime_graphs(cfg, num_vars=4, seed=21)
    mechanism = mechanisms_mod.sample_regime_mechanisms(cfg, graph, seed=22)
    sample = simulate_mod.simulate_dynscm_series(
        cfg,
        graph,
        mechanism,
        num_steps=32,
        seed=23,
    )

    assert sample.clipped
    assert sample.num_attempts == 1
    assert np.isfinite(sample.series).all()
    assert sample.max_abs_value <= cfg.max_abs_x


def test_sample_innovations_student_t_is_finite_and_scaled(dynscm_api):
    config_mod, _, _, _, simulate_mod = dynscm_api

    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "noise_family": "student_t",
            "student_df": 5,
            "noise_scale_min": 1.0,
            "noise_scale_max": 1.0,
        }
    )
    innovations, noise_scales = simulate_mod.sample_innovations(
        cfg, num_steps=200, num_vars=4, seed=31
    )

    assert innovations.shape == (200, 4)
    assert noise_scales.shape == (4,)
    assert np.isfinite(innovations).all()
    assert np.allclose(noise_scales, 1.0)


def test_validate_graph_mechanism_mismatch_raises(dynscm_api):
    config_mod, graph_mod, _, mechanisms_mod, simulate_mod = dynscm_api

    cfg_a = config_mod.DynSCMConfig.from_dict({"num_regimes": 2})
    cfg_b = config_mod.DynSCMConfig.from_dict({"num_regimes": 3})
    graph = graph_mod.sample_regime_graphs(cfg_a, num_vars=4, seed=41)
    mechanism = mechanisms_mod.sample_regime_mechanisms(
        cfg_b, graph_mod.sample_regime_graphs(cfg_b, num_vars=4, seed=42), seed=43
    )

    with pytest.raises(ValueError, match="num_regimes"):
        simulate_mod.simulate_dynscm_series(
            cfg_a, graph, mechanism, num_steps=10, seed=44
        )


def test_resolve_regime_path_validates_shape_and_bounds(dynscm_api):
    config_mod, graph_mod, _, mechanisms_mod, simulate_mod = dynscm_api

    cfg = config_mod.DynSCMConfig.from_dict(
        {"num_regimes": 2, "mechanism_type": "linear_var"}
    )
    graph = graph_mod.sample_regime_graphs(cfg, num_vars=4, seed=51)
    mechanism = mechanisms_mod.sample_regime_mechanisms(cfg, graph, seed=52)

    wrong_shape = np.zeros((5,), dtype=np.int64)
    with pytest.raises(ValueError, match="regime_path shape"):
        simulate_mod.simulate_dynscm_series(
            cfg, graph, mechanism, num_steps=10, regime_path=wrong_shape, seed=53
        )

    out_of_bounds = np.full((10,), fill_value=5, dtype=np.int64)
    with pytest.raises(ValueError, match="regime_path values"):
        simulate_mod.simulate_dynscm_series(
            cfg, graph, mechanism, num_steps=10, regime_path=out_of_bounds, seed=54
        )
