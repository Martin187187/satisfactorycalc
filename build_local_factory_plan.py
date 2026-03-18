from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Set, Iterable, Optional

from load_data import load_model, Item, Recipe

# Change this import to your optimizer filename if needed.
from solver import (
    RecipeMode,
    POWER_ITEM,
    POWERSHARD_ITEM,
    SOMERSLOOP_ITEM,
    build_recipe_modes,
)

EPS = 1e-9
SPECIAL_ITEMS = {POWER_ITEM, POWERSHARD_ITEM, SOMERSLOOP_ITEM}


# -----------------------------
# Data classes
# -----------------------------

@dataclass
class ActiveMode:
    mode_id: str
    mode: RecipeMode
    buildings: float
    produces: Dict[str, float]
    consumes: Dict[str, float]


@dataclass
class MicroNode:
    micro_id: str
    parent_mode_id: str
    mode: RecipeMode
    buildings: float
    produces: Dict[str, float]
    consumes: Dict[str, float]


@dataclass
class FlowEdge:
    item: str
    producer_id: str
    consumer_id: str
    amount: float


@dataclass
class Cluster:
    cluster_id: int
    node_ids: Set[str] = field(default_factory=set)

    produces: Dict[str, float] = field(default_factory=dict)
    consumes: Dict[str, float] = field(default_factory=dict)

    # total boundary import/export after internal cancellation
    imports: Dict[str, float] = field(default_factory=dict)
    exports: Dict[str, float] = field(default_factory=dict)

    # items consumed somewhere inside the cluster
    local_ingredient_items: Set[str] = field(default_factory=set)

    # for reporting
    flow_internalized: float = 0.0


# -----------------------------
# Generic helpers
# -----------------------------

def add_scaled(dst: Dict[str, float], src: Dict[str, float], scale: float = 1.0) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0.0) + v * scale


def subtract_scaled(dst: Dict[str, float], src: Dict[str, float], scale: float = 1.0) -> None:
    for k, v in src.items():
        dst[k] = dst.get(k, 0.0) - v * scale


def cleaned_positive_map(d: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        if v > EPS:
            out[k] = v
    return out


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def pretty_amount(x: float) -> str:
    if math.isclose(x, round(x), abs_tol=1e-9):
        return str(int(round(x)))
    return f"{x:.6f}".rstrip("0").rstrip(".")


# -----------------------------
# Rebuild active modes
# -----------------------------

def build_active_modes(recipes: List[Recipe], result_json: dict) -> List[ActiveMode]:
    modes, _augment_recipe = build_recipe_modes(recipes)
    mode_by_id = {m.mode_id: m for m in modes}

    active: List[ActiveMode] = []
    mode_usage: Dict[str, float] = result_json.get("mode_usage", {})

    for mode_id, buildings in mode_usage.items():
        buildings = float(buildings)
        if buildings <= EPS:
            continue

        mode = mode_by_id.get(mode_id)
        if mode is None:
            raise KeyError(f"Mode from result.json not found in rebuilt modes: {mode_id}")

        produces: Dict[str, float] = {}
        consumes: Dict[str, float] = {}

        for item, net_per_building in mode.item_net_per_min.items():
            total = net_per_building * buildings
            if total > EPS:
                produces[item] = total
            elif total < -EPS:
                consumes[item] = -total

        active.append(
            ActiveMode(
                mode_id=mode_id,
                mode=mode,
                buildings=buildings,
                produces=produces,
                consumes=consumes,
            )
        )

    return active


# -----------------------------
# Micro-node splitting
# -----------------------------

def split_active_modes_into_micro_nodes(
    active_modes: List[ActiveMode],
    max_item_io: float,
    ignore_items_for_chunking: Optional[Set[str]] = None,
    min_chunk_buildings: float = 1e-6,
) -> List[MicroNode]:
    """
    Split each active mode into smaller micro-nodes so no single micro-node
    has > max_item_io throughput for any tracked item.

    This makes later partitioning possible.
    """
    ignore_items_for_chunking = ignore_items_for_chunking or set()

    micro_nodes: List[MicroNode] = []

    for active in active_modes:
        tracked_amounts: List[float] = []

        for item, amt in active.produces.items():
            if item in SPECIAL_ITEMS or item in ignore_items_for_chunking:
                continue
            tracked_amounts.append(amt)

        for item, amt in active.consumes.items():
            if item in SPECIAL_ITEMS or item in ignore_items_for_chunking:
                continue
            tracked_amounts.append(amt)

        if not tracked_amounts:
            parts = 1
        else:
            max_amt = max(tracked_amounts)
            parts = max(1, math.ceil(max_amt / max_item_io))

        # If the building count is tiny, do not over-fragment.
        parts = min(parts, max(1, math.ceil(active.buildings / min_chunk_buildings)))

        for idx in range(parts):
            share = 1.0 / parts
            micro_buildings = active.buildings * share

            produces = {k: v * share for k, v in active.produces.items() if v * share > EPS}
            consumes = {k: v * share for k, v in active.consumes.items() if v * share > EPS}

            micro_nodes.append(
                MicroNode(
                    micro_id=f"{active.mode_id}#part={idx+1}/{parts}",
                    parent_mode_id=active.mode_id,
                    mode=active.mode,
                    buildings=micro_buildings,
                    produces=produces,
                    consumes=consumes,
                )
            )

    return micro_nodes


# -----------------------------
# Real flow assignment
# -----------------------------

def build_real_flows(
    micro_nodes: List[MicroNode],
    max_edge_amount: float,
) -> List[FlowEdge]:
    """
    For each item, greedily match producers to consumers with bounded edges.
    Every created edge amount <= max_edge_amount.

    This creates a real transport graph instead of only aggregate balances.
    """
    produces_by_item: Dict[str, List[Tuple[str, float]]] = {}
    consumes_by_item: Dict[str, List[Tuple[str, float]]] = {}

    for n in micro_nodes:
        for item, amt in n.produces.items():
            if item in SPECIAL_ITEMS:
                continue
            produces_by_item.setdefault(item, []).append((n.micro_id, amt))

        for item, amt in n.consumes.items():
            if item in SPECIAL_ITEMS:
                continue
            consumes_by_item.setdefault(item, []).append((n.micro_id, amt))

    flows: List[FlowEdge] = []

    for item in sorted(set(produces_by_item) & set(consumes_by_item)):
        producers = [[pid, amt] for pid, amt in produces_by_item[item] if amt > EPS]
        consumers = [[cid, amt] for cid, amt in consumes_by_item[item] if amt > EPS]

        # largest first gives fewer edges in practice
        producers.sort(key=lambda x: -x[1])
        consumers.sort(key=lambda x: -x[1])

        i = 0
        j = 0

        while i < len(producers) and j < len(consumers):
            pid, p_left = producers[i]
            cid, c_left = consumers[j]

            if pid == cid:
                # same micro-node: self-supply is not a transport edge, just skip the overlap internally
                internal = min(p_left, c_left)
                producers[i][1] -= internal
                consumers[j][1] -= internal

                if producers[i][1] <= EPS:
                    i += 1
                if consumers[j][1] <= EPS:
                    j += 1
                continue

            flow = min(producers[i][1], consumers[j][1], max_edge_amount)
            if flow > EPS:
                flows.append(
                    FlowEdge(
                        item=item,
                        producer_id=pid,
                        consumer_id=cid,
                        amount=flow,
                    )
                )
                producers[i][1] -= flow
                consumers[j][1] -= flow

            if producers[i][1] <= EPS:
                i += 1
            if consumers[j][1] <= EPS:
                j += 1

    return flows


# -----------------------------
# Cluster state
# -----------------------------

def build_node_lookup(micro_nodes: List[MicroNode]) -> Dict[str, MicroNode]:
    return {n.micro_id: n for n in micro_nodes}


def build_flow_indexes(flows: List[FlowEdge]) -> tuple[
    Dict[str, List[FlowEdge]],
    Dict[str, List[FlowEdge]],
    Dict[Tuple[str, str], float],
]:
    outgoing: Dict[str, List[FlowEdge]] = {}
    incoming: Dict[str, List[FlowEdge]] = {}
    pair_weight: Dict[Tuple[str, str], float] = {}

    for e in flows:
        outgoing.setdefault(e.producer_id, []).append(e)
        incoming.setdefault(e.consumer_id, []).append(e)

        key = (e.producer_id, e.consumer_id)
        pair_weight[key] = pair_weight.get(key, 0.0) + e.amount

    return outgoing, incoming, pair_weight


def cluster_from_nodes(
    cluster_id: int,
    node_ids: Iterable[str],
    node_lookup: Dict[str, MicroNode],
    flows: List[FlowEdge],
) -> Cluster:
    node_id_set = set(node_ids)
    c = Cluster(cluster_id=cluster_id, node_ids=node_id_set)

    for nid in node_id_set:
        node = node_lookup[nid]
        add_scaled(c.produces, node.produces, 1.0)
        add_scaled(c.consumes, node.consumes, 1.0)

    c.local_ingredient_items = {item for item, amt in c.consumes.items() if amt > EPS}

    internal_in_by_item: Dict[str, float] = {}
    internal_out_by_item: Dict[str, float] = {}

    # boundary flow accounting
    import_by_item: Dict[str, float] = {}
    export_by_item: Dict[str, float] = {}

    internalized = 0.0

    for e in flows:
        p_in = e.producer_id in node_id_set
        c_in = e.consumer_id in node_id_set

        if p_in and c_in:
            internalized += e.amount
            internal_in_by_item[e.item] = internal_in_by_item.get(e.item, 0.0) + e.amount
            internal_out_by_item[e.item] = internal_out_by_item.get(e.item, 0.0) + e.amount
        elif p_in and not c_in:
            export_by_item[e.item] = export_by_item.get(e.item, 0.0) + e.amount
        elif not p_in and c_in:
            import_by_item[e.item] = import_by_item.get(e.item, 0.0) + e.amount

    c.imports = cleaned_positive_map(import_by_item)
    c.exports = cleaned_positive_map(export_by_item)
    c.flow_internalized = internalized

    return c


def cluster_policy_violations(cluster: Cluster, max_item_io: float) -> List[str]:
    problems: List[str] = []

    for item, amt in sorted(cluster.imports.items()):
        if amt > max_item_io + EPS:
            problems.append(f"import {item} = {amt:.3f}/min > {max_item_io:.3f}/min")

    for item, amt in sorted(cluster.exports.items()):
        if amt > max_item_io + EPS:
            problems.append(f"export {item} = {amt:.3f}/min > {max_item_io:.3f}/min")

    # export only items that are not ingredients in the same cluster
    for item, amt in sorted(cluster.exports.items()):
        if amt > EPS and item in cluster.local_ingredient_items:
            problems.append(
                f"exported item {item} is also consumed locally "
                f"({amt:.3f}/min exported)"
            )

    return problems


def cluster_is_valid(cluster: Cluster, max_item_io: float) -> bool:
    return not cluster_policy_violations(cluster, max_item_io)


# -----------------------------
# Candidate adjacency
# -----------------------------

def build_neighbor_graph(flows: List[FlowEdge]) -> Dict[str, Dict[str, float]]:
    """
    Undirected weighted neighbor graph from real flows.
    """
    nbrs: Dict[str, Dict[str, float]] = {}

    for e in flows:
        if e.producer_id == e.consumer_id:
            continue

        nbrs.setdefault(e.producer_id, {})
        nbrs.setdefault(e.consumer_id, {})

        nbrs[e.producer_id][e.consumer_id] = nbrs[e.producer_id].get(e.consumer_id, 0.0) + e.amount
        nbrs[e.consumer_id][e.producer_id] = nbrs[e.consumer_id].get(e.producer_id, 0.0) + e.amount

    return nbrs


# -----------------------------
# Greedy partition building
# -----------------------------

def cluster_score(cluster: Cluster) -> float:
    """
    Favor clusters that internalize more flow and have smaller boundaries.
    """
    boundary_penalty = sum(cluster.imports.values()) + sum(cluster.exports.values())
    return cluster.flow_internalized - boundary_penalty


def greedy_partition_micro_nodes(
    micro_nodes: List[MicroNode],
    flows: List[FlowEdge],
    max_item_io: float,
) -> List[Cluster]:
    node_lookup = build_node_lookup(micro_nodes)
    neighbor_graph = build_neighbor_graph(flows)

    remaining: Set[str] = set(node_lookup.keys())
    clusters: List[Cluster] = []
    next_cluster_id = 1

    # good seeds are high-flow nodes first
    node_strength: Dict[str, float] = {nid: sum(neighbor_graph.get(nid, {}).values()) for nid in node_lookup}
    seed_order = sorted(node_lookup.keys(), key=lambda nid: (-node_strength[nid], nid))

    while remaining:
        seed = next((nid for nid in seed_order if nid in remaining), None)
        if seed is None:
            seed = next(iter(remaining))

        current_node_ids = {seed}
        remaining.remove(seed)

        current_cluster = cluster_from_nodes(next_cluster_id, current_node_ids, node_lookup, flows)

        # Singleton must itself be valid. It should be, because micro-splitting uses the same cap.
        # But if not, we still keep it; there is nothing smaller to split here.
        improved = True
        while improved:
            improved = False

            candidate_weights: Dict[str, float] = {}

            for nid in current_node_ids:
                for nb, w in neighbor_graph.get(nid, {}).items():
                    if nb in remaining:
                        candidate_weights[nb] = candidate_weights.get(nb, 0.0) + w

            # best-connected neighbors first
            candidate_order = sorted(candidate_weights.items(), key=lambda kv: (-kv[1], kv[0]))

            best_trial: Optional[Cluster] = None
            best_candidate: Optional[str] = None
            best_gain = -1e18

            for candidate_id, _weight in candidate_order:
                trial_nodes = set(current_node_ids)
                trial_nodes.add(candidate_id)

                trial_cluster = cluster_from_nodes(next_cluster_id, trial_nodes, node_lookup, flows)
                if not cluster_is_valid(trial_cluster, max_item_io):
                    continue

                gain = cluster_score(trial_cluster) - cluster_score(current_cluster)
                if gain > best_gain + EPS:
                    best_gain = gain
                    best_trial = trial_cluster
                    best_candidate = candidate_id

            if best_trial is not None and best_candidate is not None:
                current_cluster = best_trial
                current_node_ids.add(best_candidate)
                remaining.remove(best_candidate)
                improved = True

        clusters.append(current_cluster)
        next_cluster_id += 1

    return clusters


# -----------------------------
# Optional post-merge compaction
# -----------------------------

def try_compact_clusters(
    clusters: List[Cluster],
    node_lookup: Dict[str, MicroNode],
    flows: List[FlowEdge],
    max_item_io: float,
) -> List[Cluster]:
    """
    After initial clustering, try to merge whole clusters if still valid.
    """
    changed = True
    current = list(clusters)

    while changed:
        changed = False

        # cluster adjacency from flows
        node_to_cluster: Dict[str, int] = {}
        for idx, c in enumerate(current):
            for nid in c.node_ids:
                node_to_cluster[nid] = idx

        cluster_edges: Dict[Tuple[int, int], float] = {}
        for e in flows:
            a = node_to_cluster[e.producer_id]
            b = node_to_cluster[e.consumer_id]
            if a == b:
                continue
            key = (min(a, b), max(a, b))
            cluster_edges[key] = cluster_edges.get(key, 0.0) + e.amount

        if not cluster_edges:
            break

        best_pair: Optional[Tuple[int, int]] = None
        best_trial: Optional[Cluster] = None
        best_gain = -1e18

        for (a, b), _w in sorted(cluster_edges.items(), key=lambda kv: -kv[1]):
            merged_nodes = set(current[a].node_ids) | set(current[b].node_ids)
            trial = cluster_from_nodes(
                cluster_id=min(current[a].cluster_id, current[b].cluster_id),
                node_ids=merged_nodes,
                node_lookup=node_lookup,
                flows=flows,
            )
            if not cluster_is_valid(trial, max_item_io):
                continue

            gain = cluster_score(trial) - (cluster_score(current[a]) + cluster_score(current[b]))
            if gain > best_gain + EPS:
                best_gain = gain
                best_pair = (a, b)
                best_trial = trial

        if best_pair is not None and best_trial is not None:
            a, b = best_pair
            new_current: List[Cluster] = []
            for idx, c in enumerate(current):
                if idx not in {a, b}:
                    new_current.append(c)
            new_current.append(best_trial)
            current = new_current
            changed = True

    # Renumber for nicer output
    renumbered: List[Cluster] = []
    for idx, c in enumerate(sorted(current, key=lambda x: cluster_score(x), reverse=True), start=1):
        renumbered.append(
            Cluster(
                cluster_id=idx,
                node_ids=set(c.node_ids),
                produces=dict(c.produces),
                consumes=dict(c.consumes),
                imports=dict(c.imports),
                exports=dict(c.exports),
                local_ingredient_items=set(c.local_ingredient_items),
                flow_internalized=c.flow_internalized,
            )
        )
    return renumbered


# -----------------------------
# Output helpers
# -----------------------------

def cluster_to_dict(cluster: Cluster, node_lookup: Dict[str, MicroNode], max_show_nodes: int | None = None) -> dict:
    nodes_sorted = sorted(
        cluster.node_ids,
        key=lambda nid: (
            node_lookup[nid].mode.recipe_name,
            node_lookup[nid].micro_id,
        ),
    )

    if max_show_nodes is not None:
        nodes_sorted = nodes_sorted[:max_show_nodes]

    violations = cluster_policy_violations(cluster, max_item_io=float("inf"))
    # do not check max here again; output actual structural policy issue separately
    structural_violations = [
        v for v in violations
        if "also consumed locally" in v
    ]

    return {
        "cluster_id": cluster.cluster_id,
        "score": cluster_score(cluster),
        "flow_internalized": cluster.flow_internalized,
        "node_count": len(cluster.node_ids),
        "imports": dict(sorted(cluster.imports.items())),
        "exports": dict(sorted(cluster.exports.items())),
        "local_ingredient_items": sorted(cluster.local_ingredient_items),
        "policy_violations": structural_violations,
        "nodes": [
            {
                "micro_id": nid,
                "parent_mode_id": node_lookup[nid].parent_mode_id,
                "recipe_name": node_lookup[nid].mode.recipe_name,
                "recipe_class": node_lookup[nid].mode.recipe_class,
                "building_name": node_lookup[nid].mode.building_name,
                "building_class": node_lookup[nid].mode.building_class,
                "buildings": node_lookup[nid].buildings,
                "clock": node_lookup[nid].mode.clock,
                "power_shards": node_lookup[nid].mode.power_shards,
                "somersloops": node_lookup[nid].mode.somersloops,
            }
            for nid in nodes_sorted
        ],
    }


def build_output_payload(
    clusters: List[Cluster],
    node_lookup: Dict[str, MicroNode],
    max_item_io: float,
) -> dict:
    invalid_clusters = []
    valid_clusters = 0

    cluster_payloads = []
    for c in sorted(clusters, key=lambda cl: cluster_score(cl), reverse=True):
        violations = cluster_policy_violations(c, max_item_io)
        if violations:
            invalid_clusters.append({"cluster_id": c.cluster_id, "violations": violations})
        else:
            valid_clusters += 1

        cluster_payloads.append(cluster_to_dict(c, node_lookup))

    return {
        "cluster_count": len(clusters),
        "valid_cluster_count": valid_clusters,
        "invalid_cluster_count": len(invalid_clusters),
        "max_item_io": max_item_io,
        "clusters": cluster_payloads,
        "invalid_clusters": invalid_clusters,
    }


def print_summary(clusters: List[Cluster], node_lookup: Dict[str, MicroNode], max_item_io: float, top_n: int = 20) -> None:
    clusters_sorted = sorted(clusters, key=lambda c: cluster_score(c), reverse=True)

    valid_count = sum(1 for c in clusters_sorted if cluster_is_valid(c, max_item_io))
    print(f"Clusters created: {len(clusters_sorted)}")
    print(f"Valid clusters:   {valid_count}")
    print(f"Invalid clusters: {len(clusters_sorted) - valid_count}")
    print()

    for c in clusters_sorted[:top_n]:
        violations = cluster_policy_violations(c, max_item_io)

        print(f"=== Cluster {c.cluster_id} ===")
        print(f"Nodes:            {len(c.node_ids)}")
        print(f"Internalized:     {c.flow_internalized:.3f}")
        print(f"Score:            {cluster_score(c):.3f}")
        print(f"Imports total:    {sum(c.imports.values()):.3f}/min")
        print(f"Exports total:    {sum(c.exports.values()):.3f}/min")
        print(f"Valid:            {'yes' if not violations else 'no'}")

        if c.imports:
            print("Top imports:")
            for item, amt in sorted(c.imports.items(), key=lambda kv: -kv[1])[:8]:
                print(f"  {item}: {amt:.3f}/min")

        if c.exports:
            print("Top exports:")
            for item, amt in sorted(c.exports.items(), key=lambda kv: -kv[1])[:8]:
                print(f"  {item}: {amt:.3f}/min")

        if violations:
            print("Violations:")
            for v in violations[:8]:
                print(f"  - {v}")

        print("Sample nodes:")
        for nid in list(sorted(c.node_ids))[:6]:
            n = node_lookup[nid]
            print(
                f"  - {n.mode.recipe_name} | buildings={pretty_amount(n.buildings)} "
                f"| micro={n.micro_id}"
            )
        print()


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Partition an optimized Satisfactory factory into local clusters with bounded "
            "per-item import/export capacity."
        )
    )
    parser.add_argument("game_json", help="Path to the game JSON used by load_data.py")
    parser.add_argument("result_json", help="Path to result.json from the optimizer")
    parser.add_argument(
        "--output-json",
        type=str,
        default="clustered_factory_partitions.json",
        help="Where to save the partition result as JSON",
    )
    parser.add_argument(
        "--max-item-io",
        type=float,
        default=200.0,
        help="Maximum allowed import/export per item per cluster",
    )
    parser.add_argument(
        "--ignore-items-for-chunking",
        nargs="*",
        default=[],
        help=(
            "Optional item classes to ignore during micro-node chunking. "
            "Useful if you want to treat something like water specially."
        ),
    )
    parser.add_argument(
        "--skip-compaction",
        action="store_true",
        help="Skip the second pass that tries to merge valid clusters together.",
    )
    parser.add_argument(
        "--summary-top",
        type=int,
        default=20,
        help="How many clusters to print in the terminal summary",
    )

    args = parser.parse_args()

    items, recipes, _recipes_by_product, _buildings_by_class = load_model(args.game_json)
    result_json = load_json(args.result_json)

    active_modes = build_active_modes(recipes, result_json)
    if not active_modes:
        raise RuntimeError("No active recipe modes found in result.json")

    ignore_items_for_chunking = set(args.ignore_items_for_chunking)

    micro_nodes = split_active_modes_into_micro_nodes(
        active_modes=active_modes,
        max_item_io=args.max_item_io,
        ignore_items_for_chunking=ignore_items_for_chunking,
    )

    flows = build_real_flows(
        micro_nodes=micro_nodes,
        max_edge_amount=args.max_item_io,
    )

    initial_clusters = greedy_partition_micro_nodes(
        micro_nodes=micro_nodes,
        flows=flows,
        max_item_io=args.max_item_io,
    )

    node_lookup = build_node_lookup(micro_nodes)

    if args.skip_compaction:
        final_clusters = initial_clusters
    else:
        final_clusters = try_compact_clusters(
            clusters=initial_clusters,
            node_lookup=node_lookup,
            flows=flows,
            max_item_io=args.max_item_io,
        )

    print_summary(
        clusters=final_clusters,
        node_lookup=node_lookup,
        max_item_io=args.max_item_io,
        top_n=args.summary_top,
    )

    payload = build_output_payload(
        clusters=final_clusters,
        node_lookup=node_lookup,
        max_item_io=args.max_item_io,
    )
    save_json(args.output_json, payload)

    print(f"Saved clustering result to: {args.output_json}")


if __name__ == "__main__":
    main()