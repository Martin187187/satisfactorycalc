from __future__ import annotations

import argparse
import html
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from load_data import load_model, Recipe, Item


POWER_ITEM = "__POWER_MW__"
POWERSHARD_ITEM = "__POWERSHARD__"
SOMERSLOOP_ITEM = "__SOMERSLOOP__"
SPECIAL_ITEMS = {POWER_ITEM, POWERSHARD_ITEM, SOMERSLOOP_ITEM}
EPS = 1e-9


def pretty_amount(x: float) -> str:
    if math.isclose(x, round(x), abs_tol=1e-9):
        return str(int(round(x)))
    return f"{x:.3f}".rstrip("0").rstrip(".")


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_mode_id(mode_id: str) -> dict[str, Any]:
    parts = mode_id.split("|")
    recipe_class = parts[0]

    result: dict[str, Any] = {
        "recipe_class": recipe_class,
        "clock": 1.0,
        "power_shards": 0,
        "somersloops_used": 0,
        "somersloops_slots": 0,
    }

    for p in parts[1:]:
        if p.startswith("clk="):
            result["clock"] = float(p[len("clk="):])
        elif p.startswith("ps="):
            result["power_shards"] = int(float(p[len("ps="):]))
        elif p.startswith("sl="):
            used, slots = p[len("sl="):].split("/", 1)
            result["somersloops_used"] = int(used)
            result["somersloops_slots"] = int(slots)

    return result


def recipe_duration_minutes(recipe: Recipe) -> float:
    duration_s = float(getattr(recipe, "duration_s", 0.0) or 0.0)
    return duration_s / 60.0 if duration_s > 0 else 0.0


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


def somersloop_output_multiplier(sloops_used: int, max_slots: int) -> float:
    if max_slots <= 0:
        return 1.0
    return 1.0 + (float(sloops_used) / float(max_slots))


def dot_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def shorten(text: str, max_len: int = 28) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def aggregate_recipe_flows(
    result_data: dict[str, Any],
    recipes: list[Recipe],
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
]:
    """
    Returns:
      recipe_inputs[recipe_class][item_class] = flow/min
      recipe_outputs[recipe_class][item_class] = flow/min
      recipe_meta[recipe_class] = aggregated metadata
    """
    recipe_by_class = {r.class_name: r for r in recipes}
    mode_usage: dict[str, float] = {
        k: float(v)
        for k, v in result_data.get("mode_usage", {}).items()
        if float(v) > EPS
    }

    recipe_inputs: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    recipe_outputs: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    recipe_meta: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for mode_id, building_count in mode_usage.items():
        parsed = parse_mode_id(mode_id)
        recipe_class = parsed["recipe_class"]
        recipe = recipe_by_class.get(recipe_class)
        if recipe is None:
            continue

        clock = float(parsed["clock"])
        power_shards = int(parsed["power_shards"])
        sloops_used = int(parsed["somersloops_used"])
        sloops_slots = int(parsed["somersloops_slots"])
        out_mult = somersloop_output_multiplier(sloops_used, sloops_slots)

        ing_rates = recipe_base_ingredient_rates_per_min(recipe)
        prod_rates = recipe_base_product_rates_per_min(recipe)

        for item_class, base_rate in ing_rates.items():
            recipe_inputs[recipe_class][item_class] += (
                base_rate * clock * building_count
            )

        for item_class, base_rate in prod_rates.items():
            recipe_outputs[recipe_class][item_class] += (
                base_rate * clock * out_mult * building_count
            )

        recipe_meta[recipe_class]["buildings"] += building_count
        recipe_meta[recipe_class]["shards"] += power_shards * building_count
        recipe_meta[recipe_class]["sloops"] += sloops_used * building_count

    return recipe_inputs, recipe_outputs, recipe_meta


def flow_penwidth(flow: float) -> str:
    width = 0.4 + min(2.0, math.log10(max(flow, 1.0)) * 0.6)
    return f"{width:.2f}"


def flow_weight(flow: float) -> int:
    if flow >= 5000:
        return 12
    if flow >= 2000:
        return 10
    if flow >= 1000:
        return 8
    if flow >= 500:
        return 6
    if flow >= 100:
        return 4
    return 2


def edge_color_for_flow(flow: float) -> str:
    if flow >= 2000:
        return "#555555"
    if flow >= 500:
        return "#777777"
    return "#999999"


def make_item_html_label(
    name: str,
    produced: float,
    consumed: float,
    leftover: float,
    sunk: float,
) -> str:
    rows = [
        f'<TR><TD BGCOLOR="#F4E7A1"><B>{html.escape(shorten(name, 32))}</B></TD></TR>',
    ]

    if produced > EPS:
        rows.append(
            f'<TR><TD ALIGN="LEFT">Produced: {html.escape(pretty_amount(produced))}/min</TD></TR>'
        )
    if consumed > EPS:
        rows.append(
            f'<TR><TD ALIGN="LEFT">Consumed: {html.escape(pretty_amount(consumed))}/min</TD></TR>'
        )
    if leftover > EPS:
        rows.append(
            f'<TR><TD ALIGN="LEFT">Leftover: {html.escape(pretty_amount(leftover))}/min</TD></TR>'
        )
    if sunk > EPS:
        rows.append(
            f'<TR><TD ALIGN="LEFT">Sunk: {html.escape(pretty_amount(sunk))}/min</TD></TR>'
        )

    return (
        '<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4" COLOR="#B9AA5A">'
        + "".join(rows)
        + "</TABLE>"
    )


def make_recipe_html_label(
    recipe_name: str,
    buildings: float,
    shards: float,
    sloops: float,
    total_in: float,
    total_out: float,
) -> str:
    rows = [
        f'<TR><TD BGCOLOR="#BFD7FF"><B>{html.escape(shorten(recipe_name, 34))}</B></TD></TR>',
        f'<TR><TD ALIGN="LEFT">Buildings: {html.escape(pretty_amount(buildings))}</TD></TR>',
    ]

    if total_in > EPS:
        rows.append(
            f'<TR><TD ALIGN="LEFT">Input flow: {html.escape(pretty_amount(total_in))}/min</TD></TR>'
        )
    if total_out > EPS:
        rows.append(
            f'<TR><TD ALIGN="LEFT">Output flow: {html.escape(pretty_amount(total_out))}/min</TD></TR>'
        )
    if shards > EPS:
        rows.append(
            f'<TR><TD ALIGN="LEFT">Power shards: {html.escape(pretty_amount(shards))}</TD></TR>'
        )
    if sloops > EPS:
        rows.append(
            f'<TR><TD ALIGN="LEFT">Somersloops: {html.escape(pretty_amount(sloops))}</TD></TR>'
        )

    return (
        '<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4" COLOR="#7AA4E8">'
        + "".join(rows)
        + "</TABLE>"
    )


def assign_layers_best_effort(
    used_items: set[str],
    used_recipes: set[str],
    recipe_inputs: dict[str, dict[str, float]],
    recipe_outputs: dict[str, dict[str, float]],
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Alternate layers:
      source items -> 0
      source recipes -> 1
      produced items -> 2
      next recipes -> 3
      ...

    Cycles are handled best-effort by iterative relaxation and fallback passes.
    """
    producers_by_item: dict[str, set[str]] = defaultdict(set)
    consumers_by_item: dict[str, set[str]] = defaultdict(set)

    for recipe_class in used_recipes:
        for item_class in recipe_inputs.get(recipe_class, {}):
            consumers_by_item[item_class].add(recipe_class)
        for item_class in recipe_outputs.get(recipe_class, {}):
            producers_by_item[item_class].add(recipe_class)

    source_recipes = {
        recipe_class
        for recipe_class in used_recipes
        if len(recipe_inputs.get(recipe_class, {})) == 0
    }

    item_level: dict[str, int] = {}
    recipe_level: dict[str, int] = {}

    for item_class in used_items:
        producers = producers_by_item.get(item_class, set())
        if not producers:
            item_level[item_class] = 0
        elif any(p in source_recipes for p in producers):
            item_level[item_class] = 0

    for recipe_class in source_recipes:
        recipe_level[recipe_class] = 1

    changed = True
    for _ in range(200):
        if not changed:
            break
        changed = False

        for recipe_class in used_recipes:
            if recipe_class in source_recipes:
                continue

            ins = recipe_inputs.get(recipe_class, {})
            if ins and all(item in item_level for item in ins):
                lvl = max(item_level[item] for item in ins) + 1
                old = recipe_level.get(recipe_class)
                if old is None or lvl < old:
                    recipe_level[recipe_class] = lvl
                    changed = True

        for item_class in used_items:
            producers = producers_by_item.get(item_class, set())
            if not producers:
                continue
            known = [recipe_level[p] + 1 for p in producers if p in recipe_level]
            if known:
                lvl = min(known)
                old = item_level.get(item_class)
                if old is None or lvl < old:
                    item_level[item_class] = lvl
                    changed = True

    for _ in range(50):
        changed = False

        for recipe_class in used_recipes:
            if recipe_class not in recipe_level:
                ins = recipe_inputs.get(recipe_class, {})
                known_in = [item_level[item] for item in ins if item in item_level]
                if known_in:
                    recipe_level[recipe_class] = max(known_in) + 1
                else:
                    recipe_level[recipe_class] = 1
                changed = True

        for item_class in used_items:
            if item_class not in item_level:
                producers = producers_by_item.get(item_class, set())
                known_prod = [recipe_level[p] + 1 for p in producers if p in recipe_level]
                if known_prod:
                    item_level[item_class] = min(known_prod)
                else:
                    item_level[item_class] = 0
                changed = True

        if not changed:
            break

    return item_level, recipe_level


def build_graph_dot(
    result_data: dict[str, Any],
    items: list[Item],
    recipes: list[Recipe],
    include_special_items: bool,
    min_edge_flow: float,
    show_edge_labels: bool,
) -> str:
    item_names = {it.class_name: it.name for it in items}
    recipe_by_class = {r.class_name: r for r in recipes}
    leftovers: dict[str, float] = {
        k: float(v) for k, v in result_data.get("leftover_items", {}).items()
    }
    sunk_items: dict[str, float] = {
        k: float(v) for k, v in result_data.get("sunk_items", {}).items()
    }

    recipe_inputs, recipe_outputs, recipe_meta = aggregate_recipe_flows(
        result_data, recipes
    )

    visible_recipe_inputs: dict[str, dict[str, float]] = defaultdict(dict)
    visible_recipe_outputs: dict[str, dict[str, float]] = defaultdict(dict)

    used_items: set[str] = set()
    used_recipes: set[str] = set()

    for recipe_class, inputs in recipe_inputs.items():
        for item_class, flow in inputs.items():
            if flow < min_edge_flow:
                continue
            if not include_special_items and item_class in SPECIAL_ITEMS:
                continue
            visible_recipe_inputs[recipe_class][item_class] = flow
            used_items.add(item_class)
            used_recipes.add(recipe_class)

    for recipe_class, outputs in recipe_outputs.items():
        for item_class, flow in outputs.items():
            if flow < min_edge_flow:
                continue
            if not include_special_items and item_class in SPECIAL_ITEMS:
                continue
            visible_recipe_outputs[recipe_class][item_class] = flow
            used_items.add(item_class)
            used_recipes.add(recipe_class)

    for item_class, amt in leftovers.items():
        if amt > EPS and (include_special_items or item_class not in SPECIAL_ITEMS):
            used_items.add(item_class)
    for item_class, amt in sunk_items.items():
        if amt > EPS and (include_special_items or item_class not in SPECIAL_ITEMS):
            used_items.add(item_class)

    item_total_in: dict[str, float] = defaultdict(float)
    item_total_out: dict[str, float] = defaultdict(float)

    for _recipe_class, inputs in visible_recipe_inputs.items():
        for item_class, flow in inputs.items():
            item_total_in[item_class] += flow

    for _recipe_class, outputs in visible_recipe_outputs.items():
        for item_class, flow in outputs.items():
            item_total_out[item_class] += flow

    item_level, recipe_level = assign_layers_best_effort(
        used_items=used_items,
        used_recipes=used_recipes,
        recipe_inputs=visible_recipe_inputs,
        recipe_outputs=visible_recipe_outputs,
    )

    rank_to_nodes: dict[int, list[str]] = defaultdict(list)
    for item_class in used_items:
        rank_to_nodes[item_level.get(item_class, 0)].append(f"item::{item_class}")
    for recipe_class in used_recipes:
        rank_to_nodes[recipe_level.get(recipe_class, 1)].append(
            f"recipe::{recipe_class}"
        )

    lines: list[str] = []
    lines.append("digraph FactoryOverview {")
    lines.append('  rankdir=TB;')
    lines.append('  layout=dot;')
    lines.append('  splines=ortho;')
    lines.append('  concentrate=false;')
    lines.append('  compound=true;')
    lines.append('  newrank=true;')
    lines.append('  ranksep=1.4 equally;')
    lines.append('  nodesep=0.7;')
    lines.append('  esep=0.8;')
    lines.append('  sep="+20,20";')
    lines.append('  margin=0.25;')
    lines.append('  ordering=out;')
    lines.append("")
    lines.append('  node [fontname="Arial", fontsize=10, margin=0.05];')
    lines.append(
        '  edge [fontname="Arial", fontsize=8, arrowsize=0.7, color="#666666"];'
    )
    lines.append("")

    lines.append("  // Item nodes")
    for item_class in sorted(
        used_items,
        key=lambda x: (item_level.get(x, 999999), item_names.get(x, x)),
    ):
        name = item_names.get(item_class, item_class)
        leftover = leftovers.get(item_class, 0.0)
        sunk = sunk_items.get(item_class, 0.0)
        produced = item_total_out.get(item_class, 0.0)
        consumed = item_total_in.get(item_class, 0.0)

        label = make_item_html_label(
            name=name,
            produced=produced,
            consumed=consumed,
            leftover=leftover,
            sunk=sunk,
        )

        lines.append(
            f'  "item::{dot_escape(item_class)}" '
            f'[shape=plain, group="lvl_{item_level.get(item_class, 0)}", label=<{label}>];'
        )
    lines.append("")

    lines.append("  // Recipe nodes")
    for recipe_class in sorted(
        used_recipes,
        key=lambda x: (
            recipe_level.get(x, 999999),
            getattr(recipe_by_class.get(x), "name", x),
        ),
    ):
        recipe = recipe_by_class.get(recipe_class)
        if recipe is None:
            continue

        meta = recipe_meta.get(recipe_class, {})
        total_in = sum(visible_recipe_inputs.get(recipe_class, {}).values())
        total_out = sum(visible_recipe_outputs.get(recipe_class, {}).values())

        label = make_recipe_html_label(
            recipe_name=getattr(recipe, "name", recipe_class),
            buildings=meta.get("buildings", 0.0),
            shards=meta.get("shards", 0.0),
            sloops=meta.get("sloops", 0.0),
            total_in=total_in,
            total_out=total_out,
        )

        lines.append(
            f'  "recipe::{dot_escape(recipe_class)}" '
            f'[shape=plain, group="lvl_{recipe_level.get(recipe_class, 1)}", label=<{label}>];'
        )
    lines.append("")

    lines.append("  // Rank buckets")
    for rank in sorted(rank_to_nodes):
        nodes = rank_to_nodes[rank]
        if not nodes:
            continue
        lines.append(f'  subgraph "rank_{rank}" {{')
        lines.append('    rank="same";')
        for node_id in nodes:
            lines.append(f'    "{dot_escape(node_id)}";')
        lines.append("  }")
    lines.append("")

    lines.append("  // Edges")
    edge_order: list[str] = []

    for recipe_class in sorted(used_recipes, key=lambda x: recipe_level.get(x, 0)):
        for item_class, flow in sorted(
            visible_recipe_inputs.get(recipe_class, {}).items(),
            key=lambda kv: (item_level.get(kv[0], 0), item_names.get(kv[0], kv[0])),
        ):
            attrs = []
            if show_edge_labels:
                attrs.append(f'label="{dot_escape(pretty_amount(flow))}/min"')
            attrs.append(f"penwidth={flow_penwidth(flow)}")
            attrs.append(f"weight={flow_weight(flow)}")
            attrs.append("minlen=1")
            attrs.append(f'color="{edge_color_for_flow(flow)}"')

            edge_order.append(
                f'  "item::{dot_escape(item_class)}" -> "recipe::{dot_escape(recipe_class)}" '
                f'[{", ".join(attrs)}];'
            )

    for recipe_class in sorted(used_recipes, key=lambda x: recipe_level.get(x, 0)):
        for item_class, flow in sorted(
            visible_recipe_outputs.get(recipe_class, {}).items(),
            key=lambda kv: (item_level.get(kv[0], 0), item_names.get(kv[0], kv[0])),
        ):
            attrs = []
            if show_edge_labels:
                attrs.append(f'label="{dot_escape(pretty_amount(flow))}/min"')
            attrs.append(f"penwidth={flow_penwidth(flow)}")
            attrs.append(f"weight={flow_weight(flow)}")
            attrs.append("minlen=1")
            attrs.append(f'color="{edge_color_for_flow(flow)}"')

            edge_order.append(
                f'  "recipe::{dot_escape(recipe_class)}" -> "item::{dot_escape(item_class)}" '
                f'[{", ".join(attrs)}];'
            )

    lines.extend(edge_order)
    lines.append("}")

    return "\n".join(lines)


def render_with_graphviz(dot_path: Path, output_format: str) -> Path:
    out_path = dot_path.with_suffix(f".{output_format}")
    subprocess.run(
        ["dot", f"-T{output_format}", str(dot_path), "-o", str(out_path)],
        check=True,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a cleaner Graphviz recipe graph."
    )
    parser.add_argument("result_json")
    parser.add_argument("game_json")
    parser.add_argument("--out-dot", default="factory_graph_clean.dot")
    parser.add_argument("--render", choices=["svg", "png"], default="")
    parser.add_argument("--include-special-items", action="store_true")
    parser.add_argument("--min-edge-flow", type=float, default=50.0)
    parser.add_argument("--show-edge-labels", action="store_true")

    args = parser.parse_args()

    result_data = load_json(args.result_json)
    items, recipes, _recipes_by_product, _buildings_by_class = load_model(
        args.game_json
    )

    dot_text = build_graph_dot(
        result_data=result_data,
        items=items,
        recipes=recipes,
        include_special_items=args.include_special_items,
        min_edge_flow=args.min_edge_flow,
        show_edge_labels=args.show_edge_labels,
    )

    out_dot = Path(args.out_dot)
    out_dot.write_text(dot_text, encoding="utf-8")
    print(f"Wrote DOT graph to: {out_dot.resolve()}")

    if args.render:
        out_rendered = render_with_graphviz(out_dot, args.render)
        print(f"Wrote rendered graph to: {out_rendered.resolve()}")


if __name__ == "__main__":
    main()