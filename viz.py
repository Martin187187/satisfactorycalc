from __future__ import annotations

import argparse
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
        k: float(v) for k, v in result_data.get("mode_usage", {}).items()
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
            recipe_inputs[recipe_class][item_class] += base_rate * clock * building_count

        for item_class, base_rate in prod_rates.items():
            recipe_outputs[recipe_class][item_class] += base_rate * clock * out_mult * building_count

        recipe_meta[recipe_class]["buildings"] += building_count
        recipe_meta[recipe_class]["shards"] += power_shards * building_count
        recipe_meta[recipe_class]["sloops"] += sloops_used * building_count

    return recipe_inputs, recipe_outputs, recipe_meta


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
    leftovers: dict[str, float] = {k: float(v) for k, v in result_data.get("leftover_items", {}).items()}
    sunk_items: dict[str, float] = {k: float(v) for k, v in result_data.get("sunk_items", {}).items()}

    recipe_inputs, recipe_outputs, recipe_meta = aggregate_recipe_flows(result_data, recipes)

    used_items: set[str] = set()
    used_recipes: set[str] = set()
    edges: list[str] = []

    for recipe_class, inputs in recipe_inputs.items():
        used_recipes.add(recipe_class)
        for item_class, flow in inputs.items():
            if flow < min_edge_flow:
                continue
            if not include_special_items and item_class in SPECIAL_ITEMS:
                continue
            used_items.add(item_class)

            attrs = []
            if show_edge_labels:
                attrs.append(f'label="{dot_escape(pretty_amount(flow))}/min"')
            attrs.append('penwidth=1.4')
            attrs_str = ", ".join(attrs)

            edges.append(
                f'  "item::{dot_escape(item_class)}" -> "recipe::{dot_escape(recipe_class)}" [{attrs_str}];'
            )

    for recipe_class, outputs in recipe_outputs.items():
        used_recipes.add(recipe_class)
        for item_class, flow in outputs.items():
            if flow < min_edge_flow:
                continue
            if not include_special_items and item_class in SPECIAL_ITEMS:
                continue
            used_items.add(item_class)

            attrs = []
            if show_edge_labels:
                attrs.append(f'label="{dot_escape(pretty_amount(flow))}/min"')
            attrs.append('penwidth=1.4')
            attrs_str = ", ".join(attrs)

            edges.append(
                f'  "recipe::{dot_escape(recipe_class)}" -> "item::{dot_escape(item_class)}" [{attrs_str}];'
            )

    lines: list[str] = []
    lines.append("digraph FactoryOverview {")
    lines.append('  rankdir=TB;')
    lines.append('  graph [splines=true, overlap=false, ranksep=0.6, nodesep=0.25, pad=0.2, concentrate=true];')
    lines.append('  node [fontname="Arial", fontsize=10];')
    lines.append('  edge [fontname="Arial", fontsize=8, arrowsize=0.7];')
    lines.append("")

    lines.append("  // Item nodes")
    for item_class in sorted(used_items):
        name = item_names.get(item_class, item_class)
        label_parts = [shorten(name, 24)]

        leftover = leftovers.get(item_class, 0.0)
        sunk = sunk_items.get(item_class, 0.0)
        if sunk > EPS:
            label_parts.append(f"sink {pretty_amount(sunk)}")
        elif leftover > EPS:
            label_parts.append(f"left {pretty_amount(leftover)}")

        label = "\\n".join(label_parts)

        lines.append(
            f'  "item::{dot_escape(item_class)}" '
            f'[shape=ellipse, style=filled, fillcolor="lightgoldenrod1", label="{dot_escape(label)}"];'
        )

    lines.append("")
    lines.append("  // Recipe nodes")
    for recipe_class in sorted(used_recipes):
        recipe = recipe_by_class.get(recipe_class)
        if recipe is None:
            continue

        meta = recipe_meta.get(recipe_class, {})
        label = "\\n".join(
            [
                shorten(getattr(recipe, "name", recipe_class), 28),
                f"bld {pretty_amount(meta.get('buildings', 0.0))}",
                f"ps {pretty_amount(meta.get('shards', 0.0))}",
                f"sl {pretty_amount(meta.get('sloops', 0.0))}",
            ]
        )

        lines.append(
            f'  "recipe::{dot_escape(recipe_class)}" '
            f'[shape=box, style="rounded,filled", fillcolor="lightblue", label="{dot_escape(label)}"];'
        )

    lines.append("")
    lines.append("  // Edges")
    lines.extend(edges)
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
    parser = argparse.ArgumentParser(description="Create a cleaner Graphviz recipe graph.")
    parser.add_argument("result_json")
    parser.add_argument("game_json")
    parser.add_argument("--out-dot", default="factory_graph_clean.dot")
    parser.add_argument("--render", choices=["svg", "png"], default="")
    parser.add_argument("--include-special-items", action="store_true")
    parser.add_argument("--min-edge-flow", type=float, default=10.0)
    parser.add_argument("--show-edge-labels", action="store_true")

    args = parser.parse_args()

    result_data = load_json(args.result_json)
    items, recipes, _recipes_by_product, _buildings_by_class = load_model(args.game_json)

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