"""Graph and config invariants for DynSCM."""

from __future__ import annotations

import numpy as np
import pytest


def _order_positions(order: np.ndarray) -> np.ndarray:
    positions = np.empty(order.shape[0], dtype=np.int64)
    positions[order] = np.arange(order.shape[0], dtype=np.int64)
    return positions


@pytest.fixture(scope="module")
def dynscm_api(dynscm_modules):
    return dynscm_modules["config"], dynscm_modules["graph"]


def test_config_from_dict_coerces_tuples_and_rng_is_deterministic(dynscm_api):
    config_mod, _ = dynscm_api

    cfg = config_mod.DynSCMConfig.from_dict(
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
    config_mod, _ = dynscm_api

    with pytest.raises(ValueError):
        config_mod.DynSCMConfig.from_dict({"horizon_choices": [1, 3, 6]})


def test_graph_sampling_is_seed_deterministic(dynscm_api):
    config_mod, graph_mod = dynscm_api

    cfg = config_mod.DynSCMConfig()
    sample_one = graph_mod.sample_regime_graphs(cfg, num_vars=8, seed=17)
    sample_two = graph_mod.sample_regime_graphs(cfg, num_vars=8, seed=17)

    assert np.array_equal(sample_one.regime_topo_orders, sample_two.regime_topo_orders)
    assert np.array_equal(
        sample_one.base_contemp_adjacency,
        sample_two.base_contemp_adjacency,
    )
    assert np.array_equal(
        sample_one.base_lagged_adjacency,
        sample_two.base_lagged_adjacency,
    )
    assert np.array_equal(
        sample_one.regime_contemp_adjacency,
        sample_two.regime_contemp_adjacency,
    )
    assert np.array_equal(
        sample_one.regime_lagged_adjacency,
        sample_two.regime_lagged_adjacency,
    )


@pytest.mark.parametrize("use_contemp_edges", [True, False])
def test_contemporaneous_edges_respect_sampled_order(
    dynscm_api,
    use_contemp_edges: bool,
):
    config_mod, graph_mod = dynscm_api
    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "use_contemp_edges": use_contemp_edges,
            "share_base_graph": False,
            "shared_order": False,
        }
    )
    sample = graph_mod.sample_regime_graphs(cfg, num_vars=8, seed=7)

    for regime_idx in range(sample.num_regimes):
        order = sample.regime_topo_orders[regime_idx]
        positions = _order_positions(order)
        adjacency = sample.regime_contemp_adjacency[regime_idx]

        if not use_contemp_edges:
            assert not adjacency.any()
            continue

        for source, target in np.argwhere(adjacency):
            assert positions[source] < positions[target]


def test_lag_edges_reference_past_steps_and_match_parent_sets(dynscm_api):
    config_mod, graph_mod = dynscm_api
    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "max_lag": 7,
            "shared_order": False,
            "share_base_graph": False,
        }
    )
    sample = graph_mod.sample_regime_graphs(cfg, num_vars=6, seed=11)

    for regime_idx in range(sample.num_regimes):
        lagged = sample.regime_lagged_adjacency[regime_idx]
        parents_by_target = sample.regime_parent_sets[regime_idx]

        for target_idx in range(sample.num_vars):
            expected_lagged = {
                (int(source), lag_idx + 1)
                for lag_idx in range(sample.max_lag)
                for source in np.flatnonzero(lagged[lag_idx, :, target_idx])
            }
            actual_lagged = {
                (int(source), int(lag))
                for source, lag in parents_by_target[target_idx]
                if lag > 0
            }

            assert actual_lagged == expected_lagged
            assert all(1 <= lag <= sample.max_lag for _, lag in actual_lagged)
