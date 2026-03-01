"""Graph sampling utilities for DynSCM regime-specific structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import numpy as np
import numpy.typing as npt

from .config import DynSCMConfig

type Int64Arr = npt.NDArray[np.int64]
type BoolArr = npt.NDArray[np.bool_]

type RegimeOrders = Annotated[Int64Arr, "shape=(K,p)"]
type SharedOrder = Annotated[Int64Arr, "shape=(p,)"]
type Adj2D = Annotated[BoolArr, "shape=(p,p)"]
type LagAdj3D = Annotated[BoolArr, "shape=(L,p,p)"]
type RegimeAdj3D = Annotated[BoolArr, "shape=(K,p,p)"]
type RegimeLagAdj4D = Annotated[BoolArr, "shape=(K,L,p,p)"]

type Parent = tuple[int, int]  # (source, lag)
type NodeParents = tuple[Parent, ...]
type RegimeParents = tuple[NodeParents, ...]
type ParentsByRegime = tuple[RegimeParents, ...]


@dataclass(frozen=True, slots=True)
class DynSCMGraphSample:
    regime_topo_orders: RegimeOrders
    shared_topo_order: SharedOrder
    base_contemp_adjacency: Adj2D
    base_lagged_adjacency: LagAdj3D
    regime_contemp_adjacency: RegimeAdj3D
    regime_lagged_adjacency: RegimeLagAdj4D
    regime_parent_sets: ParentsByRegime
    forced_target_lag_parent: npt.NDArray[np.bool_]
    forced_target_self_lag: npt.NDArray[np.bool_]
    target_lag_parent_counts_native: npt.NDArray[np.int64]
    target_lag_parent_counts_final: npt.NDArray[np.int64]
    target_native_lag1_self_edge: npt.NDArray[np.bool_]

    @property
    def num_regimes(self) -> int:
        return self.regime_topo_orders.shape[0]

    @property
    def num_vars(self) -> int:
        return self.regime_topo_orders.shape[1]

    @property
    def max_lag(self) -> int:
        return self.regime_lagged_adjacency.shape[1]

    @property
    def target_lag_parent_counts(self) -> npt.NDArray[np.int64]:
        return self.target_lag_parent_counts_final


def sample_regime_graphs(
    cfg: DynSCMConfig,
    num_vars: int,
    *,
    target_idx: int | None = None,
    seed: int | None = None,
) -> DynSCMGraphSample:
    """Sample regime-specific graphs using a base template and sparse edge flips."""

    if not cfg.num_variables_min <= num_vars <= cfg.num_variables_max:
        raise ValueError(
            "num_vars="
            f"{num_vars} is outside configured range "
            f"[{cfg.num_variables_min}, {cfg.num_variables_max}]."
        )

    rng = cfg.make_rng(seed)
    regime_topo_orders, shared_topo_order = _sample_orders(
        num_vars=num_vars,
        num_regimes=cfg.num_regimes,
        shared=cfg.shared_order,
        rng=rng,
    )

    base_contemp_adjacency = _sample_contemporaneous(
        num_vars=num_vars,
        order=shared_topo_order,
        use_contemp_edges=cfg.use_contemp_edges,
        parent_rate=cfg.contemp_parent_rate,
        max_parents=cfg.max_contemp_parents,
        rng=rng,
    )
    base_lagged_adjacency = _sample_lagged(
        cfg=cfg,
        num_vars=num_vars,
        max_lag=cfg.max_lag,
        parent_rate=cfg.lagged_parent_rate,
        edge_decay_rate=cfg.lagged_edge_decay_rate,
        base_edge_prob=cfg.base_lagged_edge_prob,
        max_parents=cfg.max_lagged_parents,
        target_idx=target_idx,
        rng=rng,
    )

    regime_contemp_adjacency = np.zeros(
        (cfg.num_regimes, num_vars, num_vars), dtype=bool
    )
    regime_lagged_adjacency = np.zeros(
        (cfg.num_regimes, cfg.max_lag, num_vars, num_vars), dtype=bool
    )
    forced_target_lag_parent = np.zeros((cfg.num_regimes,), dtype=bool)
    forced_target_self_lag = np.zeros((cfg.num_regimes,), dtype=bool)
    target_lag_parent_counts_native = np.full((cfg.num_regimes,), -1, dtype=np.int64)
    target_lag_parent_counts_final = np.full((cfg.num_regimes,), -1, dtype=np.int64)
    target_native_lag1_self_edge = np.zeros((cfg.num_regimes,), dtype=bool)

    for regime_idx in range(cfg.num_regimes):
        regime_topological_order = regime_topo_orders[regime_idx]

        if cfg.share_base_graph:
            regime_contemp = base_contemp_adjacency.copy()
            regime_lagged = base_lagged_adjacency.copy()
        else:
            regime_contemp = _sample_contemporaneous(
                num_vars=num_vars,
                order=regime_topological_order,
                use_contemp_edges=cfg.use_contemp_edges,
                parent_rate=cfg.contemp_parent_rate,
                max_parents=cfg.max_contemp_parents,
                rng=rng,
            )
            regime_lagged = _sample_lagged(
                cfg=cfg,
                num_vars=num_vars,
                max_lag=cfg.max_lag,
                parent_rate=cfg.lagged_parent_rate,
                edge_decay_rate=cfg.lagged_edge_decay_rate,
                base_edge_prob=cfg.base_lagged_edge_prob,
                max_parents=cfg.max_lagged_parents,
                target_idx=target_idx,
                rng=rng,
            )

        regime_contemp = _flip_contemporaneous(
            adjacency=regime_contemp,
            order=regime_topological_order,
            edge_add_prob=cfg.contemp_edge_add_prob,
            edge_del_prob=cfg.contemp_edge_del_prob,
            max_parents=cfg.max_contemp_parents,
            use_contemp_edges=cfg.use_contemp_edges,
            rng=rng,
        )
        regime_lagged = _flip_lagged(
            cfg=cfg,
            lagged=regime_lagged,
            edge_add_prob=cfg.lagged_edge_add_prob,
            edge_del_prob=cfg.lagged_edge_del_prob,
            max_parents=cfg.max_lagged_parents,
            rng=rng,
        )
        if target_idx is not None:
            target_lag_parent_counts_native[regime_idx] = int(
                np.sum(regime_lagged[:, :, target_idx])
            )
            target_native_lag1_self_edge[regime_idx] = bool(
                regime_lagged[0, int(target_idx), int(target_idx)]
            )
        (
            regime_lagged,
            forced_target_parent,
            forced_target_self,
        ) = _enforce_lagged_parent_invariants(
            cfg=cfg,
            lagged=regime_lagged,
            target_idx=target_idx,
            rng=rng,
        )

        regime_contemp_adjacency[regime_idx] = regime_contemp
        regime_lagged_adjacency[regime_idx] = regime_lagged
        forced_target_lag_parent[regime_idx] = forced_target_parent
        forced_target_self_lag[regime_idx] = forced_target_self
        if target_idx is not None:
            target_lag_parent_counts_final[regime_idx] = int(
                np.sum(regime_lagged[:, :, target_idx])
            )

    _validate_graph_arrays(
        orders=regime_topo_orders,
        contemp_by_regime=regime_contemp_adjacency,
        lagged_by_regime=regime_lagged_adjacency,
        use_contemp_edges=cfg.use_contemp_edges,
        max_contemp_parents=cfg.max_contemp_parents,
        max_lagged_parents=cfg.max_lagged_parents,
    )

    regime_parent_sets = _build_parent_lists(
        regime_contemp_adjacency, regime_lagged_adjacency
    )

    return DynSCMGraphSample(
        regime_topo_orders=regime_topo_orders,
        shared_topo_order=shared_topo_order,
        base_contemp_adjacency=base_contemp_adjacency,
        base_lagged_adjacency=base_lagged_adjacency,
        regime_contemp_adjacency=regime_contemp_adjacency,
        regime_lagged_adjacency=regime_lagged_adjacency,
        regime_parent_sets=regime_parent_sets,
        forced_target_lag_parent=forced_target_lag_parent,
        forced_target_self_lag=forced_target_self_lag,
        target_lag_parent_counts_native=target_lag_parent_counts_native,
        target_lag_parent_counts_final=target_lag_parent_counts_final,
        target_native_lag1_self_edge=target_native_lag1_self_edge,
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
    use_contemp_edges: bool,
    parent_rate: float,
    max_parents: int,
    rng: np.random.Generator,
) -> np.ndarray:
    adjacency = np.zeros((num_vars, num_vars), dtype=bool)
    if not use_contemp_edges or max_parents == 0:
        return adjacency

    positions = _order_positions(order)
    for target in range(num_vars):
        available = order[: positions[target]]
        if available.size == 0:
            continue
        max_parents_for_target = min(max_parents, available.size)
        parent_count = _sample_truncated_poisson(
            rng, parent_rate, max_parents_for_target
        )
        if parent_count == 0:
            continue
        parents = rng.choice(available, size=parent_count, replace=False)
        adjacency[parents, target] = True

    np.fill_diagonal(adjacency, False)
    return adjacency


def _sample_lagged(
    *,
    cfg: DynSCMConfig,
    num_vars: int,
    max_lag: int,
    parent_rate: float,
    edge_decay_rate: float,
    base_edge_prob: float,
    max_parents: int,
    target_idx: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if cfg.lagged_sampler_mode == "separated_self_cross":
        return _sample_lagged_separated_self_cross(
            cfg=cfg,
            num_vars=num_vars,
            max_lag=max_lag,
            parent_rate=parent_rate,
            edge_decay_rate=edge_decay_rate,
            base_edge_prob=base_edge_prob,
            max_parents=max_parents,
            rng=rng,
        )

    return _sample_lagged_legacy(
        cfg=cfg,
        num_vars=num_vars,
        max_lag=max_lag,
        parent_rate=parent_rate,
        edge_decay_rate=edge_decay_rate,
        base_edge_prob=base_edge_prob,
        max_parents=max_parents,
        target_idx=target_idx,
        rng=rng,
    )


def _sample_lagged_legacy(
    *,
    cfg: DynSCMConfig,
    num_vars: int,
    max_lag: int,
    parent_rate: float,
    edge_decay_rate: float,
    base_edge_prob: float,
    max_parents: int,
    target_idx: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    lagged = np.zeros((max_lag, num_vars, num_vars), dtype=bool)
    if max_parents == 0:
        return lagged

    all_sources = np.arange(num_vars, dtype=np.int64)
    for lag_idx in range(max_lag):
        decay = float(np.exp(-edge_decay_rate * lag_idx))
        lam = parent_rate * decay
        edge_prob = min(1.0, base_edge_prob * decay)

        for target in range(num_vars):
            max_parents_for_target = min(max_parents, num_vars)
            parent_count = _sample_truncated_poisson(rng, lam, max_parents_for_target)
            if parent_count == 0:
                continue

            sampled_mask = rng.random(num_vars) < edge_prob
            candidates = all_sources[sampled_mask]
            if candidates.size == 0:
                continue

            selected_count = min(parent_count, candidates.size)
            parents = rng.choice(candidates, size=selected_count, replace=False)
            lagged[lag_idx, parents, target] = True

    _bias_target_lagged_sampling(
        cfg=cfg,
        lagged=lagged,
        target_idx=target_idx,
        max_parents=max_parents,
        edge_decay_rate=edge_decay_rate,
        rng=rng,
    )
    return lagged


def _sample_lagged_separated_self_cross(
    *,
    cfg: DynSCMConfig,
    num_vars: int,
    max_lag: int,
    parent_rate: float,
    edge_decay_rate: float,
    base_edge_prob: float,
    max_parents: int,
    rng: np.random.Generator,
) -> np.ndarray:
    lagged = np.zeros((max_lag, num_vars, num_vars), dtype=bool)
    if max_parents == 0:
        return lagged

    all_sources = np.arange(num_vars, dtype=np.int64)

    for lag_idx in range(max_lag):
        self_prob = float(
            cfg.self_lag_prob * np.exp(-cfg.self_lag_decay_rate * lag_idx)
        )
        if self_prob > 0.0:
            sampled_self = rng.random(num_vars) < min(1.0, self_prob)
            for node in np.flatnonzero(sampled_self):
                lagged[lag_idx, int(node), int(node)] = True

    for lag_idx in range(max_lag):
        decay = float(np.exp(-edge_decay_rate * lag_idx))
        lam = parent_rate * decay
        edge_prob = min(1.0, base_edge_prob * decay)

        for target in range(num_vars):
            has_self = bool(lagged[lag_idx, target, target])
            max_cross_parents = max(0, min(max_parents - int(has_self), num_vars - 1))
            if max_cross_parents == 0:
                continue

            parent_count = _sample_truncated_poisson(rng, lam, max_cross_parents)
            if parent_count == 0:
                continue

            cross_sources = np.concatenate(
                (all_sources[:target], all_sources[target + 1 :])
            )
            sampled_mask = rng.random(cross_sources.size) < edge_prob
            candidates = cross_sources[sampled_mask]
            if candidates.size == 0:
                continue

            selected_count = min(parent_count, candidates.size)
            parents = rng.choice(candidates, size=selected_count, replace=False)
            lagged[lag_idx, parents, target] = True

    return lagged


def _bias_target_lagged_sampling(
    *,
    cfg: DynSCMConfig,
    lagged: np.ndarray,
    target_idx: int | None,
    max_parents: int,
    edge_decay_rate: float,
    rng: np.random.Generator,
) -> None:
    if target_idx is None or max_parents == 0:
        return

    target = int(target_idx)
    num_lags, num_vars, _ = lagged.shape

    if cfg.target_native_self_lag_prob > 0.0 and rng.random() < float(
        cfg.target_native_self_lag_prob
    ):
        lagged[0, target, target] = True

    min_target_parents = int(cfg.target_native_min_lagged_parents)
    if min_target_parents <= 0:
        return

    lag_weights = np.exp(
        -float(edge_decay_rate) * np.arange(num_lags, dtype=np.float64)
    )
    if float(lag_weights.sum()) <= 0.0:
        lag_weights = np.ones((num_lags,), dtype=np.float64)
    lag_weights = lag_weights / lag_weights.sum()

    while int(np.sum(lagged[:, :, target])) < min_target_parents:
        candidate_slots: list[tuple[int, int, float]] = []
        for lag_idx in range(num_lags):
            current_parent_count = int(np.sum(lagged[lag_idx, :, target]))
            if current_parent_count >= max_parents:
                continue
            for source in range(num_vars):
                if lagged[lag_idx, source, target]:
                    continue
                weight = float(lag_weights[lag_idx])
                if lag_idx == 0 and source == target:
                    weight *= 2.0
                candidate_slots.append((lag_idx, source, weight))

        if not candidate_slots:
            break

        probs = np.asarray(
            [weight for _, _, weight in candidate_slots], dtype=np.float64
        )
        if float(probs.sum()) <= 0.0:
            probs = np.full((len(candidate_slots),), 1.0 / len(candidate_slots))
        else:
            probs = probs / probs.sum()
        slot_idx = int(
            rng.choice(np.arange(len(candidate_slots), dtype=np.int64), p=probs)
        )
        lag_idx, source, _ = candidate_slots[slot_idx]
        lagged[lag_idx, source, target] = True


def _flip_contemporaneous(
    *,
    adjacency: np.ndarray,
    order: np.ndarray,
    edge_add_prob: float,
    edge_del_prob: float,
    max_parents: int,
    use_contemp_edges: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    matrix = adjacency.copy()
    np.fill_diagonal(matrix, False)
    if not use_contemp_edges or max_parents == 0:
        return np.zeros_like(matrix, dtype=bool)

    delete_mask = (rng.random(matrix.shape) < edge_del_prob) & matrix
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
            if rng.random() < edge_add_prob:
                matrix[source, target] = True

    for target in range(num_vars):
        incoming = np.flatnonzero(matrix[:, target])
        if incoming.size <= max_parents:
            continue
        keep = set(rng.choice(incoming, size=max_parents, replace=False).tolist())
        for source in incoming.tolist():
            if source not in keep:
                matrix[source, target] = False

    np.fill_diagonal(matrix, False)
    return matrix


def _flip_lagged(
    *,
    cfg: DynSCMConfig,
    lagged: np.ndarray,
    edge_add_prob: float,
    edge_del_prob: float,
    max_parents: int,
    rng: np.random.Generator,
) -> np.ndarray:
    output = lagged.copy()
    if max_parents == 0:
        return np.zeros_like(output, dtype=bool)

    num_lags, num_vars, _ = output.shape
    for lag_idx in range(num_lags):
        matrix = output[lag_idx]
        mutable_mask = np.ones_like(matrix, dtype=bool)
        if cfg.lagged_sampler_mode == "separated_self_cross":
            np.fill_diagonal(mutable_mask, False)
        delete_mask = (rng.random(matrix.shape) < edge_del_prob) & matrix & mutable_mask
        matrix[delete_mask] = False

        add_candidates = ~matrix
        if cfg.lagged_sampler_mode == "separated_self_cross":
            add_candidates &= mutable_mask
        add_mask = (rng.random(matrix.shape) < edge_add_prob) & add_candidates
        matrix[add_mask] = True

        for target in range(num_vars):
            incoming = np.flatnonzero(matrix[:, target])
            if incoming.size <= max_parents:
                continue
            if cfg.lagged_sampler_mode == "separated_self_cross" and target in incoming:
                keep: set[int] = {int(target)}
                remaining = np.asarray(
                    [source for source in incoming.tolist() if int(source) != target],
                    dtype=np.int64,
                )
                remaining_keep = max(0, max_parents - 1)
                if remaining.size > remaining_keep:
                    keep.update(
                        map(
                            int,
                            rng.choice(
                                remaining,
                                size=remaining_keep,
                                replace=False,
                            ).tolist(),
                        )
                    )
                else:
                    keep.update(map(int, remaining.tolist()))
            else:
                keep = set(
                    rng.choice(incoming, size=max_parents, replace=False).tolist()
                )
            for source in incoming:
                if source not in keep:
                    matrix[source, target] = False

        output[lag_idx] = matrix

    return output


def _enforce_lagged_parent_invariants(
    *,
    cfg: DynSCMConfig,
    lagged: np.ndarray,
    target_idx: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool, bool]:
    output = lagged.copy()
    num_lags, num_vars, _ = output.shape
    required_targets: set[int] = set()
    if cfg.enforce_all_nodes_lagged_parent:
        required_targets.update(range(num_vars))
    if cfg.enforce_target_lagged_parent and target_idx is not None:
        required_targets.add(int(target_idx))

    forced_target_parent = False
    forced_target_self = False
    if (
        target_idx is not None
        and (
            cfg.target_self_lag_min_budget_fraction is not None
            or cfg.target_self_lag_abs_min is not None
        )
        and not bool(output[0, int(target_idx), int(target_idx)])
    ):
        output[0, int(target_idx), int(target_idx)] = True
        forced_target_parent = True
        forced_target_self = True

    for target in sorted(required_targets):
        if np.any(output[:, :, target]):
            continue
        use_self_edge = bool(
            target_idx is not None
            and target == target_idx
            and (
                cfg.force_target_self_lag_if_parentless
                or cfg.enforce_target_lagged_parent
            )
        )
        source = int(target) if use_self_edge else int(rng.integers(0, num_vars))
        output[0, source, target] = True
        if target_idx is not None and target == target_idx:
            forced_target_parent = True
            forced_target_self = bool(source == target)
    return output, forced_target_parent, forced_target_self


def _validate_graph_arrays(
    *,
    orders: np.ndarray,
    contemp_by_regime: np.ndarray,
    lagged_by_regime: np.ndarray,
    use_contemp_edges: bool,
    max_contemp_parents: int,
    max_lagged_parents: int,
) -> None:
    num_regimes, num_vars = orders.shape
    if contemp_by_regime.shape != (num_regimes, num_vars, num_vars):
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

        contemp = contemp_by_regime[regime_idx]
        if np.diag(contemp).any():
            raise RuntimeError("Contemporaneous adjacency must not contain self loops.")

        if use_contemp_edges:
            positions = _order_positions(order)
            edges = np.argwhere(contemp)
            for source, target in edges:
                if positions[source] >= positions[target]:
                    raise RuntimeError(
                        "Contemporaneous edge violates topological order "
                        f"(source={source}, target={target})."
                    )
            indegrees = contemp.sum(axis=0)
            if (indegrees > max_contemp_parents).any():
                raise RuntimeError("Contemporaneous indegree cap violated.")
        elif contemp.any():
            raise RuntimeError(
                "Contemporaneous edges present although use_contemp_edges=False."
            )

        lagged = lagged_by_regime[regime_idx]
        indegrees_by_lag = lagged.sum(axis=1)
        if (indegrees_by_lag > max_lagged_parents).any():
            raise RuntimeError(
                "Lagged indegree cap violated for at least one lag/target."
            )


def _build_parent_lists(
    contemp_by_regime: np.ndarray,
    lagged_by_regime: np.ndarray,
) -> ParentsByRegime:
    num_regimes, num_vars, _ = contemp_by_regime.shape
    max_lag = lagged_by_regime.shape[1]

    parents_all: list[tuple[tuple[tuple[int, int], ...], ...]] = []
    for regime_idx in range(num_regimes):
        regime_parents: list[tuple[tuple[int, int], ...]] = []

        contemp = contemp_by_regime[regime_idx]
        lagged = lagged_by_regime[regime_idx]

        for target in range(num_vars):
            target_parents: list[tuple[int, int]] = []

            for source in np.flatnonzero(contemp[:, target]):
                target_parents.append((int(source), 0))
            for lag_idx in range(max_lag):
                for source in np.flatnonzero(lagged[lag_idx, :, target]):
                    target_parents.append((int(source), lag_idx + 1))

            regime_parents.append(tuple(target_parents))

        parents_all.append(tuple(regime_parents))

    return tuple(parents_all)
