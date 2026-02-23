"""Concise tests for DynSCM mechanisms."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def dynscm_api(dynscm_modules):
    return (
        dynscm_modules["config"],
        dynscm_modules["graph"],
        dynscm_modules["stability"],
        dynscm_modules["mechanisms"],
    )


def test_sample_regime_mechanisms_is_seed_deterministic(dynscm_api):
    config_mod, graph_mod, _, mechanisms_mod = dynscm_api
    cfg = config_mod.DynSCMConfig()
    graph = graph_mod.sample_regime_graphs(cfg, num_vars=7, seed=5)

    m1 = mechanisms_mod.sample_regime_mechanisms(cfg, graph, seed=13)
    m2 = mechanisms_mod.sample_regime_mechanisms(cfg, graph, seed=13)

    assert m1.mechanism_type == m2.mechanism_type
    assert np.array_equal(m1.lag_coeffs, m2.lag_coeffs)
    assert np.array_equal(m1.contemp_coeffs, m2.contemp_coeffs)
    assert np.array_equal(m1.residual_input_weights, m2.residual_input_weights)
    assert np.array_equal(m1.residual_biases, m2.residual_biases)
    assert np.array_equal(m1.residual_output_weights, m2.residual_output_weights)
    assert np.array_equal(
        m1.residual_lipschitz_upper_bounds, m2.residual_lipschitz_upper_bounds
    )


def test_linear_var_mode_has_no_residual_and_matches_linear_response(dynscm_api):
    config_mod, graph_mod, stability_mod, mechanisms_mod = dynscm_api
    cfg = config_mod.DynSCMConfig.from_dict({"mechanism_type": "linear_var"})
    graph = graph_mod.sample_regime_graphs(cfg, num_vars=6, seed=7)
    stability = stability_mod.sample_stable_coefficients(cfg, graph, seed=8)
    mechanism = mechanisms_mod.sample_regime_mechanisms(
        cfg, graph, stability_sample=stability, seed=9
    )

    assert not mechanism.has_residual
    assert mechanism.residual_num_features == 0
    assert np.array_equal(mechanism.lag_coeffs, stability.lag_coeffs)
    assert np.array_equal(
        mechanism.contemp_coeffs,
        stability.contemp_coeffs,
    )

    lagged_history = np.random.default_rng(0).normal(size=(cfg.max_lag, 6))
    direct = np.einsum("ls,lst->t", lagged_history, mechanism.lag_coeffs[0])
    evaluated = mechanisms_mod.evaluate_lagged_mechanism(
        mechanism,
        regime_idx=0,
        lagged_history=lagged_history,
    )
    assert np.allclose(evaluated, direct)


def test_linear_plus_residual_respects_lipschitz_cap(dynscm_api):
    config_mod, graph_mod, _, mechanisms_mod = dynscm_api
    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "mechanism_type": "linear_plus_residual",
            "residual_num_features": 7,
            "residual_lipschitz_max": 0.05,
        }
    )
    graph = graph_mod.sample_regime_graphs(cfg, num_vars=5, seed=11)
    mechanism = mechanisms_mod.sample_regime_mechanisms(cfg, graph, seed=12)

    assert mechanism.has_residual
    assert mechanism.residual_num_features == cfg.residual_num_features
    assert mechanism.residual_input_weights.shape == (
        cfg.num_regimes,
        5,
        cfg.residual_num_features,
        cfg.max_lag * 5,
    )
    assert np.all(
        mechanism.residual_lipschitz_upper_bounds <= cfg.residual_lipschitz_max + 1e-8
    )

    lagged_history = np.random.default_rng(1).normal(size=(cfg.max_lag, 5))
    lagged_response = mechanisms_mod.evaluate_lagged_mechanism(
        mechanism,
        regime_idx=1,
        lagged_history=lagged_history,
    )
    cont_effect = mechanisms_mod.evaluate_contemporaneous_effect(
        mechanism,
        regime_idx=1,
        current_state=np.zeros((5,), dtype=np.float64),
    )
    assert lagged_response.shape == (5,)
    assert cont_effect.shape == (5,)
