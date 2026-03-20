from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from load_data import load_model, Item, Recipe


EPS = 1e-9

POWER_ITEM = "__POWER_MW__"
POWERSHARD_ITEM = "__POWERSHARD__"
SOMERSLOOP_ITEM = "__SOMERSLOOP__"

SPECIAL_ITEMS = {POWER_ITEM, POWERSHARD_ITEM, SOMERSLOOP_ITEM}

PRODUCTION_POWER_EXPONENT = 1.3219280948873624

CLOCK_OPTIONS = (
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
    2.25,
    2.50,
)

SOMERSLOOP_SLOT_OVERRIDES: dict[str, int] = {
    "Build_SmelterMk1_C": 1,
    "Build_ConstructorMk1_C": 1,

    "Build_FoundryMk1_C": 2,
    "Build_AssemblerMk1_C": 2,
    "Build_OilRefinery_C": 2,
    "Build_Blender_C": 2,
    "Build_Converter_C": 2,

    "Build_ManufacturerMk1_C": 4,
    "Build_HadronCollider_C": 4,
    "Build_ParticleAccelerator_C": 4,
    "Build_QuantumEncoder_C": 4,

    "Build_Packager_C": 0,

    "POWER": 0,
    "AUG_POWER": 0,
}


@dataclass(frozen=True, slots=True)
class RecipeMode:
    mode_id: str
    recipe_class: str
    recipe_name: str
    building_class: str
    building_name: str
    clock: float
    power_shards: int
    somersloops: int
    somersloop_slots: int
    output_multiplier: float
    power_multiplier: float
    item_net_per_min: dict[str, float]
    power_net_mw: float


@dataclass
class VirtualNode:
    node_id: str
    mode_id: str
    recipe_name: str
    building_name: str
    building_count: float
    item_net_per_min: dict[str, float]

    @property
    def total_input(self) -> float:
        return sum(-v for k, v in self.item_net_per_min.items() if k not in SPECIAL_ITEMS and v < -EPS)

    @property
    def total_output(self) -> float:
        return sum(v for k, v in self.item_net_per_min.items() if k not in SPECIAL_ITEMS and v > EPS)

    @property
    def weight(self) -> float:
        return max(self.total_input, self.total_output)


@dataclass
class Cluster:
    cluster_id: str
    node_ids: set[str] = field(default_factory=set)
    weight: float = 0.0

    def copy(self) -> "Cluster":
        return Cluster(
            cluster_id=self.cluster_id,
            node_ids=set(self.node_ids),
            weight=self.weight,
        )


def pretty_amount(x: float) -> str:
    if math.isclose(x, round(x), abs_tol=1e-9):
        return str(int(round(x)))
    return f"{x:.6f}".rstrip("0").rstrip(".")


def pick_primary_building(recipe: Recipe):
    buildings = getattr(recipe, "produced_in_buildings", ()) or ()
    if not buildings:
        return None

    for b in buildings:
        cls = getattr(b, "class_name", "")
        if cls.startswith("Build_") or cls in {"POWER", "AUG_POWER"}:
            return b

    return buildings[0]


def recipe_duration_minutes(recipe: Recipe) -> float:
    duration_s = float(getattr(recipe, "duration_s", 0.0) or 0.0)
    if duration_s <= 0:
        return 0.0
    return duration_s / 60.0


def recipe_base_ingredient_rates_per_min(recipe: Recipe) -> dict[str, float]:
    duration_min = recipe_duration_minutes(recipe)
    if duration_min <= 0:
        return {}

    rates: dict[str, float] = {}
    for cls, amt in recipe.ingredients:
        rates[cls] = rates.get(cls, 0.0) + float(amt) / duration_min
    return rates


def recipe_base_product_rates_per_min(recipe: Recipe) -> dict[str, float]:
    duration_min = recipe_duration_minutes(recipe)
    if duration_min <= 0:
        return {}

    rates: dict[str, float] = {}
    for cls, amt in recipe.products:
        rates[cls] = rates.get(cls, 0.0) + float(amt) / duration_min
    return rates


def is_power_generator_building(recipe: Recipe) -> bool:
    building = pick_primary_building(recipe)
    if building is None:
        return False
    return float(getattr(building, "power_consumption", 0.0) or 0.0) < 0.0


def is_augment_recipe(recipe: Recipe) -> bool:
    building = pick_primary_building(recipe)
    if building is None:
        return False

    cls = getattr(building, "class_name", "")
    grid_mult = float(getattr(building, "power_grid_multiplier", 1.0) or 1.0)
    return cls == "AUG_POWER" or grid_mult > 1.0 + EPS


def is_raw_resource_generator(recipe: Recipe) -> bool:
    building = pick_primary_building(recipe)
    if building is None:
        return False

    cls = getattr(building, "class_name", "")
    name = (getattr(building, "name", "") or "").lower()

    raw_generator_classes = {
        "Build_MinerMk1_C",
        "Build_MinerMk2_C",
        "Build_MinerMk3_C",
        "Build_OilPump_C",
        "Build_WaterPump_C",
        "Build_FrackingExtractor_C",
        "Build_GeneratorGeoThermal_C",
    }

    if cls in raw_generator_classes:
        return True

    raw_keywords = (
        "miner",
        "oil extractor",
        "water extractor",
        "fracking extractor",
        "resource well extractor",
        "geothermal",
    )
    return any(k in name for k in raw_keywords)


def infer_somersloop_slots(recipe: Recipe) -> int:
    building = pick_primary_building(recipe)
    if building is None:
        return 0

    cls = getattr(building, "class_name", "")
    if cls in SOMERSLOOP_SLOT_OVERRIDES:
        return SOMERSLOOP_SLOT_OVERRIDES[cls]

    name = (getattr(building, "name", "") or "").lower()

    if "packager" in name:
        return 0
    if "manufacturer" in name or "encoder" in name or "accelerator" in name:
        return 4
    if "assembler" in name or "refinery" in name or "foundry" in name or "blender" in name or "converter" in name:
        return 2
    if "constructor" in name or "smelter" in name:
        return 1

    return 0


def max_power_shards_for_clock(clock: float) -> int:
    if clock <= 1.0 + EPS:
        return 0
    if clock <= 1.5 + EPS:
        return 1
    if clock <= 2.0 + EPS:
        return 2
    return 3


def somersloop_output_multiplier(sloops_used: int, max_slots: int) -> float:
    if max_slots <= 0:
        return 1.0
    return 1.0 + (float(sloops_used) / float(max_slots))


def somersloop_power_multiplier(sloops_used: int, max_slots: int) -> float:
    out_mult = somersloop_output_multiplier(sloops_used, max_slots)
    return out_mult * out_mult


def effective_shard_slot_size(recipe: Recipe) -> int:
    building = pick_primary_building(recipe)
    if building is None:
        return 0

    shard_slots = int(getattr(building, "production_shard_slot_size", 0) or 0)
    if is_raw_resource_generator(recipe):
        return max(3, shard_slots)
    return shard_slots


def get_clock_options_for_recipe(recipe: Recipe) -> list[float]:
    shard_slots = effective_shard_slot_size(recipe)
    if shard_slots <= 0:
        return [1.0]

    result: list[float] = []
    seen = set()

    for clock in CLOCK_OPTIONS:
        if max_power_shards_for_clock(clock) <= shard_slots and clock not in seen:
            seen.add(clock)
            result.append(clock)

    if 1.0 not in seen:
        result.insert(0, 1.0)

    return result


def recipe_mode_power_mw(recipe: Recipe, clock: float, sloops_used: int, max_sloops: int) -> float:
    building = pick_primary_building(recipe)
    if building is None:
        return 0.0

    base_power = float(getattr(building, "power_consumption", 0.0) or 0.0)

    if base_power < 0.0:
        return (-base_power) * clock

    clock_power_mult = clock ** PRODUCTION_POWER_EXPONENT
    amp_power_mult = somersloop_power_multiplier(sloops_used, max_sloops)
    return -(base_power * clock_power_mult * amp_power_mult)


def build_recipe_modes(recipes: list[Recipe]) -> tuple[list[RecipeMode], Recipe | None]:
    modes: list[RecipeMode] = []
    augment_recipe: Recipe | None = None

    for rec in recipes:
        building = pick_primary_building(rec)
        if building is None:
            continue

        if is_augment_recipe(rec):
            augment_recipe = rec
            continue

        base_ing_rates = recipe_base_ingredient_rates_per_min(rec)
        base_prod_rates = recipe_base_product_rates_per_min(rec)

        max_sloops = 0 if is_power_generator_building(rec) else infer_somersloop_slots(rec)
        clock_options = get_clock_options_for_recipe(rec)
        shard_slot_size = effective_shard_slot_size(rec)
        raw_generator = is_raw_resource_generator(rec)

        if not base_ing_rates and not base_prod_rates and abs(recipe_mode_power_mw(rec, 1.0, 0, max_sloops)) <= EPS:
            continue

        for clock in clock_options:
            shards_needed = max_power_shards_for_clock(clock)
            if shards_needed > shard_slot_size:
                continue

            sloop_values = [0] if max_sloops <= 0 else list(range(0, max_sloops + 1))

            for sloops_used in sloop_values:
                out_mult = somersloop_output_multiplier(sloops_used, max_sloops)
                pwr_mult = somersloop_power_multiplier(sloops_used, max_sloops)

                net: dict[str, float] = {}

                for cls, amt_per_min in base_ing_rates.items():
                    if raw_generator:
                        scaled = -amt_per_min
                    else:
                        scaled = -amt_per_min * clock

                    if abs(scaled) > EPS:
                        net[cls] = net.get(cls, 0.0) + scaled

                for cls, amt_per_min in base_prod_rates.items():
                    scaled = amt_per_min * clock * out_mult
                    if abs(scaled) > EPS:
                        net[cls] = net.get(cls, 0.0) + scaled

                power_net = recipe_mode_power_mw(rec, clock, sloops_used, max_sloops)

                mode_id = (
                    f"{rec.class_name}"
                    f"|clk={clock:.2f}"
                    f"|ps={shards_needed}"
                    f"|sl={sloops_used}/{max_sloops}"
                )

                modes.append(
                    RecipeMode(
                        mode_id=mode_id,
                        recipe_class=rec.class_name,
                        recipe_name=getattr(rec, "name", rec.class_name),
                        building_class=getattr(building, "class_name", "Unknown"),
                        building_name=getattr(building, "name", getattr(building, "class_name", "Unknown")),
                        clock=clock,
                        power_shards=shards_needed,
                        somersloops=sloops_used,
                        somersloop_slots=max_sloops,
                        output_multiplier=out_mult,
                        power_multiplier=pwr_mult,
                        item_net_per_min=net,
                        power_net_mw=power_net,
                    )
                )

    return modes, augment_recipe


def load_result_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def build_virtual_nodes(
    result_data: dict,
    modes_by_id: dict[str, RecipeMode],
    cluster_capacity: float,
    min_replicas: int = 1,
) -> list[VirtualNode]:
    nodes: list[VirtualNode] = []

    mode_usage: dict[str, float] = result_data.get("mode_usage", {}) or {}

    for mode_id, count in mode_usage.items():
        if count <= EPS:
            continue

        mode = modes_by_id[mode_id]

        scaled_net = {k: v * count for k, v in mode.item_net_per_min.items()}
        total_input = sum(-v for k, v in scaled_net.items() if k not in SPECIAL_ITEMS and v < -EPS)
        total_output = sum(v for k, v in scaled_net.items() if k not in SPECIAL_ITEMS and v > EPS)
        mode_weight = max(total_input, total_output)

        replicas = max(min_replicas, int(math.ceil(mode_weight / max(cluster_capacity, 1.0))))
        replicas = max(1, replicas)

        for i in range(replicas):
            frac = 1.0 / replicas
            node_id = f"{mode_id}#part={i + 1}/{replicas}"
            node_net = {k: v * count * frac for k, v in mode.item_net_per_min.items()}

            nodes.append(
                VirtualNode(
                    node_id=node_id,
                    mode_id=mode_id,
                    recipe_name=mode.recipe_name,
                    building_name=mode.building_name,
                    building_count=count * frac,
                    item_net_per_min=node_net,
                )
            )

    return nodes


def build_item_flow_edges(nodes: list[VirtualNode]) -> dict[tuple[str, str, str], float]:
    """
    Estimate producer->consumer flows by item using proportional allocation.

    Returns:
        (src_node_id, dst_node_id, item_class) -> amount_per_min
    """
    flow_edges: dict[tuple[str, str, str], float] = {}

    all_items = sorted({k for n in nodes for k in n.item_net_per_min.keys()})

    for item in all_items:
        if item in SPECIAL_ITEMS:
            continue

        producers: list[tuple[VirtualNode, float]] = []
        consumers: list[tuple[VirtualNode, float]] = []

        for n in nodes:
            amt = n.item_net_per_min.get(item, 0.0)
            if amt > EPS:
                producers.append((n, amt))
            elif amt < -EPS:
                consumers.append((n, -amt))

        if not producers or not consumers:
            continue

        total_prod = sum(v for _, v in producers)
        total_cons = sum(v for _, v in consumers)
        matched_flow = min(total_prod, total_cons)

        if matched_flow <= EPS:
            continue

        for p_node, p_amt in producers:
            p_share = p_amt / total_prod
            for c_node, c_amt in consumers:
                c_share = c_amt / total_cons
                f = matched_flow * p_share * c_share
                if f > EPS:
                    key = (p_node.node_id, c_node.node_id, item)
                    flow_edges[key] = flow_edges.get(key, 0.0) + f

    return flow_edges


def cluster_pair_internal_flow(
    a: Cluster,
    b: Cluster,
    flow_edges: dict[tuple[str, str, str], float],
) -> float:
    total = 0.0
    a_nodes = a.node_ids
    b_nodes = b.node_ids

    for (src, dst, _item), amt in flow_edges.items():
        if amt <= EPS:
            continue
        if (src in a_nodes and dst in b_nodes) or (src in b_nodes and dst in a_nodes):
            total += amt

    return total


def merge_clusters(a: Cluster, b: Cluster, new_id: str) -> Cluster:
    return Cluster(
        cluster_id=new_id,
        node_ids=a.node_ids | b.node_ids,
        weight=a.weight + b.weight,
    )


def greedy_merge_clusters(
    nodes: list[VirtualNode],
    flow_edges: dict[tuple[str, str, str], float],
    cluster_capacity: float,
    overflow_tolerance: float,
    capacity_penalty: float,
) -> list[Cluster]:
    clusters: list[Cluster] = [
        Cluster(cluster_id=f"C{i+1}", node_ids={n.node_id}, weight=n.weight)
        for i, n in enumerate(nodes)
    ]

    next_cluster_idx = len(clusters) + 1

    while True:
        best_pair: tuple[int, int] | None = None
        best_score = 0.0

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                a = clusters[i]
                b = clusters[j]

                internalized = cluster_pair_internal_flow(a, b, flow_edges)
                merged_weight = a.weight + b.weight

                allowed_weight = cluster_capacity * (1.0 + overflow_tolerance)
                overflow = max(0.0, merged_weight - allowed_weight)
                penalty = overflow * capacity_penalty

                score = internalized - penalty

                if score > best_score + 1e-9:
                    best_score = score
                    best_pair = (i, j)

        if best_pair is None:
            break

        i, j = best_pair
        a = clusters[i]
        b = clusters[j]
        merged = merge_clusters(a, b, f"C{next_cluster_idx}")
        next_cluster_idx += 1

        new_clusters: list[Cluster] = []
        for idx, c in enumerate(clusters):
            if idx not in {i, j}:
                new_clusters.append(c)
        new_clusters.append(merged)
        clusters = new_clusters

    clusters.sort(key=lambda c: (-c.weight, c.cluster_id))
    for i, c in enumerate(clusters, start=1):
        c.cluster_id = f"C{i}"

    return clusters


def cluster_import_export_summary(
    clusters: list[Cluster],
    nodes_by_id: dict[str, VirtualNode],
    flow_edges: dict[tuple[str, str, str], float],
) -> dict[str, dict]:
    node_to_cluster: dict[str, str] = {}
    for c in clusters:
        for nid in c.node_ids:
            node_to_cluster[nid] = c.cluster_id

    result: dict[str, dict] = {
        c.cluster_id: {
            "internal_flow": 0.0,
            "imports_from_clusters": {},
            "exports_to_clusters": {},
            "import_items_from_clusters": {},
            "export_items_to_clusters": {},
            "mode_entries": [],
            "total_weight": c.weight,
        }
        for c in clusters
    }

    for c in clusters:
        entries = []
        for nid in sorted(c.node_ids):
            n = nodes_by_id[nid]
            entries.append(
                {
                    "node_id": n.node_id,
                    "mode_id": n.mode_id,
                    "recipe_name": n.recipe_name,
                    "building_name": n.building_name,
                    "building_count": n.building_count,
                    "weight": n.weight,
                }
            )
        entries.sort(key=lambda x: (-x["weight"], x["recipe_name"], x["node_id"]))
        result[c.cluster_id]["mode_entries"] = entries

    for (src, dst, item), amt in flow_edges.items():
        if amt <= EPS:
            continue

        c_src = node_to_cluster[src]
        c_dst = node_to_cluster[dst]

        if c_src == c_dst:
            result[c_src]["internal_flow"] += amt
            continue

        result[c_src]["exports_to_clusters"][c_dst] = (
            result[c_src]["exports_to_clusters"].get(c_dst, 0.0) + amt
        )
        result[c_dst]["imports_from_clusters"][c_src] = (
            result[c_dst]["imports_from_clusters"].get(c_src, 0.0) + amt
        )

        export_items = result[c_src]["export_items_to_clusters"].setdefault(c_dst, {})
        export_items[item] = export_items.get(item, 0.0) + amt

        import_items = result[c_dst]["import_items_from_clusters"].setdefault(c_src, {})
        import_items[item] = import_items.get(item, 0.0) + amt

    return result


def cluster_item_balance(
    cluster: Cluster,
    nodes_by_id: dict[str, VirtualNode],
) -> dict[str, float]:
    net: dict[str, float] = {}

    for nid in cluster.node_ids:
        n = nodes_by_id[nid]
        for item, amt in n.item_net_per_min.items():
            if item in SPECIAL_ITEMS:
                continue
            net[item] = net.get(item, 0.0) + amt

    return {k: v for k, v in net.items() if abs(v) > 1e-7}


def build_output_json(
    clusters: list[Cluster],
    nodes_by_id: dict[str, VirtualNode],
    flow_edges: dict[tuple[str, str, str], float],
) -> dict:
    summary = cluster_import_export_summary(clusters, nodes_by_id, flow_edges)

    out_clusters = []
    for c in clusters:
        balance = cluster_item_balance(c, nodes_by_id)

        total_imports = sorted(
            summary[c.cluster_id]["imports_from_clusters"].items(),
            key=lambda kv: (-kv[1], kv[0]),
        )
        total_exports = sorted(
            summary[c.cluster_id]["exports_to_clusters"].items(),
            key=lambda kv: (-kv[1], kv[0]),
        )

        import_items_sorted = {
            other_cluster: dict(
                sorted(item_map.items(), key=lambda kv: (-kv[1], kv[0]))
            )
            for other_cluster, item_map in sorted(
                summary[c.cluster_id]["import_items_from_clusters"].items(),
                key=lambda kv: kv[0],
            )
        }

        export_items_sorted = {
            other_cluster: dict(
                sorted(item_map.items(), key=lambda kv: (-kv[1], kv[0]))
            )
            for other_cluster, item_map in sorted(
                summary[c.cluster_id]["export_items_to_clusters"].items(),
                key=lambda kv: kv[0],
            )
        }

        out_clusters.append(
            {
                "cluster_id": c.cluster_id,
                "weight": c.weight,
                "internal_flow": summary[c.cluster_id]["internal_flow"],
                "mode_entries": summary[c.cluster_id]["mode_entries"],
                "net_item_balance_per_min": dict(sorted(balance.items())),
                "imports_from_clusters": {k: v for k, v in total_imports},
                "exports_to_clusters": {k: v for k, v in total_exports},
                "import_items_from_clusters": import_items_sorted,
                "export_items_to_clusters": export_items_sorted,
            }
        )

    node_to_cluster: dict[str, str] = {}
    for c in clusters:
        for nid in c.node_ids:
            node_to_cluster[nid] = c.cluster_id

    cluster_edge_flow: dict[tuple[str, str], float] = {}
    cluster_edge_items: dict[tuple[str, str], dict[str, float]] = {}

    for (src, dst, item), amt in flow_edges.items():
        c_src = node_to_cluster[src]
        c_dst = node_to_cluster[dst]
        if c_src == c_dst or amt <= EPS:
            continue

        pair = (c_src, c_dst)
        cluster_edge_flow[pair] = cluster_edge_flow.get(pair, 0.0) + amt

        item_map = cluster_edge_items.setdefault(pair, {})
        item_map[item] = item_map.get(item, 0.0) + amt

    inter_cluster_edges = []
    for (src, dst), amt in sorted(cluster_edge_flow.items(), key=lambda kv: (-kv[1], kv[0])):
        item_map = cluster_edge_items[(src, dst)]
        inter_cluster_edges.append(
            {
                "from_cluster": src,
                "to_cluster": dst,
                "flow_per_min": amt,
                "items_per_min": dict(sorted(item_map.items(), key=lambda kv: (-kv[1], kv[0]))),
            }
        )

    return {
        "cluster_count": len(clusters),
        "clusters": out_clusters,
        "inter_cluster_edges": inter_cluster_edges,
    }


def save_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=False)
    print(f"Saved cluster output to: {path}")


def print_cluster_summary(
    cluster_data: dict,
    items_by_class: dict[str, Item],
    top_modes: int,
    top_links: int,
    top_items_per_link: int,
) -> None:
    print("=== CLUSTER SUMMARY ===")
    print(f"Cluster count: {cluster_data['cluster_count']}")
    print()

    for cluster in cluster_data["clusters"]:
        print(cluster["cluster_id"])
        print(f"  weight:         {cluster['weight']:.3f}")
        print(f"  internal flow:  {cluster['internal_flow']:.3f}")

        if cluster["imports_from_clusters"]:
            print("  imports:")
            for cid, amt in list(cluster["imports_from_clusters"].items())[:top_links]:
                print(f"    {cid}: {amt:.3f}/min")
                per_item = cluster["import_items_from_clusters"].get(cid, {})
                for item_cls, item_amt in list(per_item.items())[:top_items_per_link]:
                    item_name = items_by_class.get(item_cls).name if item_cls in items_by_class else item_cls
                    print(f"      {item_name} ({item_cls}): {item_amt:.3f}/min")

        if cluster["exports_to_clusters"]:
            print("  exports:")
            for cid, amt in list(cluster["exports_to_clusters"].items())[:top_links]:
                print(f"    {cid}: {amt:.3f}/min")
                per_item = cluster["export_items_to_clusters"].get(cid, {})
                for item_cls, item_amt in list(per_item.items())[:top_items_per_link]:
                    item_name = items_by_class.get(item_cls).name if item_cls in items_by_class else item_cls
                    print(f"      {item_name} ({item_cls}): {item_amt:.3f}/min")

        if cluster["mode_entries"]:
            print("  main modes:")
            for entry in cluster["mode_entries"][:top_modes]:
                print(
                    f"    {entry['recipe_name']} | "
                    f"{entry['building_name']} | "
                    f"buildings={pretty_amount(entry['building_count'])} | "
                    f"weight={entry['weight']:.3f}"
                )

        if cluster["net_item_balance_per_min"]:
            print("  net balance:")
            sorted_balance = sorted(
                cluster["net_item_balance_per_min"].items(),
                key=lambda kv: (-abs(kv[1]), kv[0]),
            )
            for item_cls, amt in sorted_balance[:top_items_per_link]:
                item_name = items_by_class.get(item_cls).name if item_cls in items_by_class else item_cls
                direction = "export surplus" if amt > 0 else "import demand"
                print(f"    {item_name} ({item_cls}): {amt:.3f}/min [{direction}]")

        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read an optimizer result JSON, rebuild the active production graph, "
            "and partition it into capacity-based production clusters with per-item "
            "import/export breakdown."
        )
    )
    parser.add_argument("json_path", help="Path to the game JSON consumed by load_data.py")
    parser.add_argument("result_json", help="Path to the optimizer result JSON")
    parser.add_argument(
        "--output-json",
        type=str,
        default="clustered_result.json",
        help="Where to save the cluster JSON",
    )
    parser.add_argument(
        "--cluster-capacity",
        type=float,
        default=5000.0,
        help="Target max throughput weight per cluster",
    )
    parser.add_argument(
        "--overflow-tolerance",
        type=float,
        default=0.15,
        help="Allow clusters to exceed target capacity by this fraction before penalties kick in",
    )
    parser.add_argument(
        "--capacity-penalty",
        type=float,
        default=10.0,
        help="Penalty per unit of cluster overflow during merge",
    )
    parser.add_argument("--top-modes", type=int, default=8)
    parser.add_argument("--top-links", type=int, default=5)
    parser.add_argument("--top-items-per-link", type=int, default=5)

    args = parser.parse_args()

    result_data = load_result_json(args.result_json)
    items, recipes, _recipes_by_product, _buildings_by_class = load_model(args.json_path)
    items_by_class = {it.class_name: it for it in items}

    modes, _augment_recipe = build_recipe_modes(recipes)
    modes_by_id = {m.mode_id: m for m in modes}

    missing_mode_ids = [
        mode_id for mode_id in (result_data.get("mode_usage", {}) or {}).keys()
        if mode_id not in modes_by_id
    ]
    if missing_mode_ids:
        raise RuntimeError(
            "Some mode_ids from result JSON could not be rebuilt from the game data.\n"
            f"First missing mode_id: {missing_mode_ids[0]}"
        )

    virtual_nodes = build_virtual_nodes(
        result_data=result_data,
        modes_by_id=modes_by_id,
        cluster_capacity=args.cluster_capacity,
    )

    if not virtual_nodes:
        raise RuntimeError("No active production modes found in result JSON.")

    nodes_by_id = {n.node_id: n for n in virtual_nodes}
    flow_edges = build_item_flow_edges(virtual_nodes)

    clusters = greedy_merge_clusters(
        nodes=virtual_nodes,
        flow_edges=flow_edges,
        cluster_capacity=args.cluster_capacity,
        overflow_tolerance=args.overflow_tolerance,
        capacity_penalty=args.capacity_penalty,
    )

    cluster_data = build_output_json(
        clusters=clusters,
        nodes_by_id=nodes_by_id,
        flow_edges=flow_edges,
    )

    save_json(args.output_json, cluster_data)
    print_cluster_summary(
        cluster_data=cluster_data,
        items_by_class=items_by_class,
        top_modes=args.top_modes,
        top_links=args.top_links,
        top_items_per_link=args.top_items_per_link,
    )


if __name__ == "__main__":
    main()