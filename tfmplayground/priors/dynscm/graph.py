"""Graph sampling utilities for DynSCM regime-specific structures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DynSCMConfig

ParentsByRegime = tuple[tuple[tuple[tuple[int, int], ...], ...], ...]


@dataclass(frozen=True, slots=True)
class DynSCMGraphSample:
    """Container for base and per-regime dynamic graph structures."""

    num_vars: int
    num_regimes: int
    max_lag: int
    regime_topological_orders: np.ndarray  # (K, p), one topological order per regime.
    shared_topological_order: np.ndarray  # (p,), shared topological order template.
    base_contemporaneous_adjacency: np.ndarray  # (p, p), base within-time adjacency.
    base_lagged_adjacency: np.ndarray  # (L, p, p), base lagged adjacency by lag.
    regime_contemporaneous_adjacency: (
        np.ndarray
    )  # (K, p, p), within-time adjacency by regime.
    regime_lagged_adjacency: (
        np.ndarray
    )  # (K, L, p, p), lagged adjacency by regime and lag.
    regime_parent_sets: (
        ParentsByRegime  # Parent sets by regime/target as (source, lag).
    )


def sample_regime_graphs(
    cfg: DynSCMConfig,
    num_vars: int,
    *,
    seed: int | None = None,
) -> DynSCMGraphSample:
    """Sample regime-specific graphs using a base template and sparse edge flips."""

    if not cfg.p_raw_min <= num_vars <= cfg.p_raw_max:
        raise ValueError(
            f"num_vars={num_vars} is outside configured range [{cfg.p_raw_min}, {cfg.p_raw_max}]."
        )

    rng = cfg.make_rng(seed)
    regime_topological_orders, shared_topological_order = _sample_orders(
        num_vars=num_vars,
        num_regimes=cfg.num_regimes,
        shared=cfg.shared_order,
        rng=rng,
    )

    base_contemporaneous_adjacency = _sample_contemporaneous(
        num_vars=num_vars,
        order=shared_topological_order,
        use_contemporaneous=cfg.use_contemporaneous,
        lambda_0=cfg.lambda_0,
        indegree_cap=cfg.dmax_0,
        rng=rng,
    )
    base_lagged_adjacency = _sample_lagged(
        num_vars=num_vars,
        max_lag=cfg.max_lag,
        lambda_lag=cfg.lambda_lag,
        lag_decay_gamma=cfg.lag_decay_gamma,
        base_lag_prob=cfg.base_lag_prob,
        indegree_cap=cfg.dmax_lag,
        rng=rng,
    )

    regime_contemporaneous_adjacency = np.zeros(
        (cfg.num_regimes, num_vars, num_vars), dtype=bool
    )
    regime_lagged_adjacency = np.zeros(
        (cfg.num_regimes, cfg.max_lag, num_vars, num_vars), dtype=bool
    )

    for regime_idx in range(cfg.num_regimes):
        regime_topological_order = regime_topological_orders[regime_idx]

        if cfg.share_base_graph:
            regime_contemporaneous = base_contemporaneous_adjacency.copy()
            regime_lagged = base_lagged_adjacency.copy()
        else:
            regime_contemporaneous = _sample_contemporaneous(
                num_vars=num_vars,
                order=regime_topological_order,
                use_contemporaneous=cfg.use_contemporaneous,
                lambda_0=cfg.lambda_0,
                indegree_cap=cfg.dmax_0,
                rng=rng,
            )
            regime_lagged = _sample_lagged(
                num_vars=num_vars,
                max_lag=cfg.max_lag,
                lambda_lag=cfg.lambda_lag,
                lag_decay_gamma=cfg.lag_decay_gamma,
                base_lag_prob=cfg.base_lag_prob,
                indegree_cap=cfg.dmax_lag,
                rng=rng,
            )

        regime_contemporaneous = _flip_contemporaneous(
            adjacency=regime_contemporaneous,
            order=regime_topological_order,
            q_add=cfg.q_add_0,
            q_del=cfg.q_del_0,
            indegree_cap=cfg.dmax_0,
            use_contemporaneous=cfg.use_contemporaneous,
            rng=rng,
        )
        regime_lagged = _flip_lagged(
            lagged=regime_lagged,
            q_add=cfg.q_add_lag,
            q_del=cfg.q_del_lag,
            indegree_cap=cfg.dmax_lag,
            rng=rng,
        )

        regime_contemporaneous_adjacency[regime_idx] = regime_contemporaneous
        regime_lagged_adjacency[regime_idx] = regime_lagged

    _validate_graph_arrays(
        orders=regime_topological_orders,
        contemporaneous_by_regime=regime_contemporaneous_adjacency,
        lagged_by_regime=regime_lagged_adjacency,
        use_contemporaneous=cfg.use_contemporaneous,
        dmax_0=cfg.dmax_0,
        dmax_lag=cfg.dmax_lag,
    )

    regime_parent_sets = _build_parent_lists(
        regime_contemporaneous_adjacency, regime_lagged_adjacency
    )

    return DynSCMGraphSample(
        num_vars=num_vars,
        num_regimes=cfg.num_regimes,
        max_lag=cfg.max_lag,
        regime_topological_orders=regime_topological_orders,
        shared_topological_order=shared_topological_order,
        base_contemporaneous_adjacency=base_contemporaneous_adjacency,
        base_lagged_adjacency=base_lagged_adjacency,
        regime_contemporaneous_adjacency=regime_contemporaneous_adjacency,
        regime_lagged_adjacency=regime_lagged_adjacency,
        regime_parent_sets=regime_parent_sets,
    )


def _sample_orders(
    *,
    num_vars: int,
    num_regimes: int,
    shared: bool,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    base_order = rng.permutation(num_vars).astype(np.int64)
    if shared:
        orders = np.tile(base_order[None, :], (num_regimes, 1))
    else:
        orders = np.stack(
            [rng.permutation(num_vars).astype(np.int64) for _ in range(num_regimes)],
            axis=0,
        )
    return orders, base_order


def _sample_truncated_poisson(rng: np.random.Generator, lam: float, cap: int) -> int:
    if cap <= 0 or lam <= 0.0:
        return 0
    return int(min(rng.poisson(lam=lam), cap))


def _order_positions(order: np.ndarray) -> np.ndarray:
    positions = np.empty(order.shape[0], dtype=np.int64)
    positions[order] = np.arange(order.shape[0], dtype=np.int64)
    return positions


def _sample_contemporaneous(
    *,
    num_vars: int,
    order: np.ndarray,
    use_contemporaneous: bool,
    lambda_0: float,
    indegree_cap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    adjacency = np.zeros((num_vars, num_vars), dtype=bool)
    if not use_contemporaneous or indegree_cap == 0:
        return adjacency

    positions = _order_positions(order)
    for target in range(num_vars):
        available = order[: positions[target]]
        if available.size == 0:
            continue
        max_parents = min(indegree_cap, available.size)
        parent_count = _sample_truncated_poisson(rng, lambda_0, max_parents)
        if parent_count == 0:
            continue
        parents = rng.choice(available, size=parent_count, replace=False)
        adjacency[parents, target] = True

    np.fill_diagonal(adjacency, False)
    return adjacency


def _sample_lagged(
    *,
    num_vars: int,
    max_lag: int,
    lambda_lag: float,
    lag_decay_gamma: float,
    base_lag_prob: float,
    indegree_cap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    lagged = np.zeros((max_lag, num_vars, num_vars), dtype=bool)
    if indegree_cap == 0:
        return lagged

    all_sources = np.arange(num_vars, dtype=np.int64)
    for lag_idx in range(max_lag):
        decay = float(np.exp(-lag_decay_gamma * lag_idx))
        lam = lambda_lag * decay
        edge_prob = min(1.0, base_lag_prob * decay)

        for target in range(num_vars):
            max_parents = min(indegree_cap, num_vars)
            parent_count = _sample_truncated_poisson(rng, lam, max_parents)
            if parent_count == 0:
                continue

            sampled_mask = rng.random(num_vars) < edge_prob
            candidates = all_sources[sampled_mask]
            if candidates.size == 0:
                continue

            selected_count = min(parent_count, candidates.size)
            parents = rng.choice(candidates, size=selected_count, replace=False)
            lagged[lag_idx, parents, target] = True

    return lagged


def _flip_contemporaneous(
    *,
    adjacency: np.ndarray,
    order: np.ndarray,
    q_add: float,
    q_del: float,
    indegree_cap: int,
    use_contemporaneous: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    matrix = adjacency.copy()
    np.fill_diagonal(matrix, False)
    if not use_contemporaneous or indegree_cap == 0:
        return np.zeros_like(matrix, dtype=bool)

    delete_mask = (rng.random(matrix.shape) < q_del) & matrix
    matrix[delete_mask] = False

    positions = _order_positions(order)
    num_vars = matrix.shape[0]
    for source in range(num_vars):
        for target in range(num_vars):
            if source == target:
                continue
            if positions[source] >= positions[target]:
                continue
            if matrix[source, target]:
                continue
            if rng.random() < q_add:
                matrix[source, target] = True

    for target in range(num_vars):
        incoming = np.flatnonzero(matrix[:, target])
        if incoming.size <= indegree_cap:
            continue
        keep = set(rng.choice(incoming, size=indegree_cap, replace=False).tolist())
        for source in incoming:
            if source not in keep:
                matrix[source, target] = False

    np.fill_diagonal(matrix, False)
    return matrix


def _flip_lagged(
    *,
    lagged: np.ndarray,
    q_add: float,
    q_del: float,
    indegree_cap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    output = lagged.copy()
    if indegree_cap == 0:
        return np.zeros_like(output, dtype=bool)

    num_lags, num_vars, _ = output.shape
    for lag_idx in range(num_lags):
        matrix = output[lag_idx]
        delete_mask = (rng.random(matrix.shape) < q_del) & matrix
        matrix[delete_mask] = False

        add_candidates = ~matrix
        add_mask = (rng.random(matrix.shape) < q_add) & add_candidates
        matrix[add_mask] = True

        for target in range(num_vars):
            incoming = np.flatnonzero(matrix[:, target])
            if incoming.size <= indegree_cap:
                continue
            keep = set(rng.choice(incoming, size=indegree_cap, replace=False).tolist())
            for source in incoming:
                if source not in keep:
                    matrix[source, target] = False

        output[lag_idx] = matrix

    return output


def _validate_graph_arrays(
    *,
    orders: np.ndarray,
    contemporaneous_by_regime: np.ndarray,
    lagged_by_regime: np.ndarray,
    use_contemporaneous: bool,
    dmax_0: int,
    dmax_lag: int,
) -> None:
    num_regimes, num_vars = orders.shape
    if contemporaneous_by_regime.shape != (num_regimes, num_vars, num_vars):
        raise RuntimeError("Invalid contemporaneous_by_regime shape.")
    if lagged_by_regime.shape[0] != num_regimes or lagged_by_regime.shape[2:] != (
        num_vars,
        num_vars,
    ):
        raise RuntimeError("Invalid lagged_by_regime shape.")

    for regime_idx in range(num_regimes):
        order = orders[regime_idx]
        if np.unique(order).size != num_vars:
            raise RuntimeError("Order is not a valid permutation.")

        contemporaneous = contemporaneous_by_regime[regime_idx]
        if np.diag(contemporaneous).any():
            raise RuntimeError("Contemporaneous adjacency must not contain self loops.")

        if use_contemporaneous:
            positions = _order_positions(order)
            edges = np.argwhere(contemporaneous)
            for source, target in edges:
                if positions[source] >= positions[target]:
                    raise RuntimeError(
                        "Contemporaneous edge violates topological order "
                        f"(source={source}, target={target})."
                    )
            indegrees = contemporaneous.sum(axis=0)
            if (indegrees > dmax_0).any():
                raise RuntimeError("Contemporaneous indegree cap violated.")
        elif contemporaneous.any():
            raise RuntimeError(
                "Contemporaneous edges present although use_contemporaneous=False."
            )

        lagged = lagged_by_regime[regime_idx]
        indegrees_by_lag = lagged.sum(axis=1)
        if (indegrees_by_lag > dmax_lag).any():
            raise RuntimeError(
                "Lagged indegree cap violated for at least one lag/target."
            )


def _build_parent_lists(
    contemporaneous_by_regime: np.ndarray,
    lagged_by_regime: np.ndarray,
) -> ParentsByRegime:
    num_regimes, num_vars, _ = contemporaneous_by_regime.shape
    max_lag = lagged_by_regime.shape[1]

    parents_all: list[tuple[tuple[tuple[int, int], ...], ...]] = []
    for regime_idx in range(num_regimes):
        regime_parents: list[tuple[tuple[int, int], ...]] = []

        contemporaneous = contemporaneous_by_regime[regime_idx]
        lagged = lagged_by_regime[regime_idx]

        for target in range(num_vars):
            target_parents: list[tuple[int, int]] = []

            for source in np.flatnonzero(contemporaneous[:, target]):
                target_parents.append((int(source), 0))
            for lag_idx in range(max_lag):
                for source in np.flatnonzero(lagged[lag_idx, :, target]):
                    target_parents.append((int(source), lag_idx + 1))

            regime_parents.append(tuple(target_parents))

        parents_all.append(tuple(regime_parents))

    return tuple(parents_all)
