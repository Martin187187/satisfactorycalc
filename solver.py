from __future__ import annotations

import argparse
from dataclasses import dataclass
import math

from scipy.optimize import linprog

from load_data import load_model, Item, Recipe


EPS = 1e-9
POWER_ITEM = "__POWER_MJ__"


@dataclass(frozen=True, slots=True)
class SolveResult:
    total_score: float
    recipe_runs: dict[str, float]
    sunk_items: dict[str, float]


@dataclass(frozen=True, slots=True)
class RecipeEnergyStat:
    recipe_class: str
    recipe_name: str
    building_name: str
    building_class: str
    runs: float
    power_mw: float
    duration_s: float
    energy_per_run_mj: float
    total_energy_mj: float


def parse_supply_args(values: list[str]) -> dict[str, float]:
    """
    Parse CLI values like:
        Desc_OreIron_C=10000
        Desc_OreCopper_C=2500
        __POWER_MJ__=500000
    """
    supplies: dict[str, float] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Supply must be in ITEM=AMOUNT form, got: {raw!r}")
        k, v = raw.split("=", 1)
        k = k.strip()
        if not k:
            raise ValueError(f"Empty item class in supply: {raw!r}")
        try:
            amt = float(v.strip())
        except ValueError as e:
            raise ValueError(f"Invalid amount in supply {raw!r}") from e
        if amt < 0:
            raise ValueError(f"Supply amount must be non-negative: {raw!r}")
        supplies[k] = supplies.get(k, 0.0) + amt
    return supplies


def pick_primary_building(recipe: Recipe):
    """
    Prefer real factory buildings over workshop/workbench helper components.
    """
    buildings = getattr(recipe, "produced_in_buildings", ()) or ()
    if not buildings:
        return None

    for b in buildings:
        if getattr(b, "class_name", "").startswith("Build_") or getattr(b, "class_name", "") == "POWER":
            return b

    return buildings[0]


def recipe_power_net_mj(recipe: Recipe) -> float:
    """
    Net batch energy contribution of one recipe run in MJ.

    Positive  => produces energy
    Negative  => consumes energy

    Building power is MW = MJ/s, so:
        energy_per_run_mj = power_mw * duration_s

    Normal manufacturers have positive power consumption -> negative net energy.
    Generators have negative power consumption -> positive net energy.
    """
    building = pick_primary_building(recipe)
    if building is None:
        return 0.0

    power_mw = float(getattr(building, "power_consumption", 0.0) or 0.0)
    duration_s = float(getattr(recipe, "duration_s", 0.0) or 0.0)

    # Resource net convention:
    # positive => produced
    # negative => consumed
    return -(power_mw * duration_s)


def build_item_maps(items: list[Item], recipes: list[Recipe], base_supplies: dict[str, float]):
    """
    Build:
      - items_by_class
      - full item set including items only mentioned in recipes/supplies
      - synthetic POWER_ITEM for energy accounting
    """
    items_by_class: dict[str, Item] = {it.class_name: it for it in items}

    all_item_classes = set(items_by_class.keys())
    all_item_classes.update(base_supplies.keys())
    all_item_classes.add(POWER_ITEM)

    for rec in recipes:
        for cls, _ in rec.ingredients:
            all_item_classes.add(cls)
        for cls, _ in rec.products:
            all_item_classes.add(cls)

        power_net_mj = recipe_power_net_mj(rec)
        if abs(power_net_mj) > EPS:
            all_item_classes.add(POWER_ITEM)

    for cls in all_item_classes:
        if cls not in items_by_class:
            items_by_class[cls] = Item(
                class_name=cls,
                name="Power (MJ)" if cls == POWER_ITEM else cls,
                sink_points=0,
            )

    return items_by_class, sorted(all_item_classes)


def recipe_net_map(recipe: Recipe) -> dict[str, float]:
    """
    Net production per single recipe execution:
      positive => produced
      negative => consumed

    Includes synthetic POWER_ITEM in MJ.
    """
    net: dict[str, float] = {}

    for cls, amt in recipe.products:
        net[cls] = net.get(cls, 0.0) + float(amt)

    for cls, amt in recipe.ingredients:
        net[cls] = net.get(cls, 0.0) - float(amt)

    power_net_mj = recipe_power_net_mj(recipe)
    if abs(power_net_mj) > EPS:
        net[POWER_ITEM] = net.get(POWER_ITEM, 0.0) + power_net_mj

    return net


def solve_max_sink_score(
    items: list[Item],
    recipes: list[Recipe],
    base_supplies: dict[str, float],
) -> SolveResult:
    items_by_class, item_classes = build_item_maps(items, recipes, base_supplies)

    sinkable_items = [cls for cls in item_classes if items_by_class[cls].sink_points > 0]
    sink_idx = {cls: k for k, cls in enumerate(sinkable_items)}

    n_recipes = len(recipes)
    n_sink = len(sinkable_items)
    n_vars = n_recipes + n_sink

    # scipy.linprog minimizes
    c = [0.0] * n_vars
    for cls, k in sink_idx.items():
        c[n_recipes + k] = -float(items_by_class[cls].sink_points)

    A_ub: list[list[float]] = []
    b_ub: list[float] = []

    recipe_nets = [recipe_net_map(r) for r in recipes]

    for cls in item_classes:
        row = [0.0] * n_vars

        for j, net in enumerate(recipe_nets):
            amt = net.get(cls, 0.0)
            if abs(amt) > EPS:
                row[j] = -amt

        sk = sink_idx.get(cls)
        if sk is not None:
            row[n_recipes + sk] = 1.0

        A_ub.append(row)
        b_ub.append(float(base_supplies.get(cls, 0.0)))

    bounds = [(0.0, None)] * n_vars

    res = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )

    if res.status == 3:
        raise RuntimeError(
            "Optimization is unbounded. "
            "This usually means there is a profitable production cycle "
            "that can generate infinite sink value from the available model."
        )

    if not res.success:
        raise RuntimeError(f"Optimization failed: {res.message}")

    x = res.x
    recipe_runs: dict[str, float] = {}
    sunk_items: dict[str, float] = {}

    for j, recipe in enumerate(recipes):
        val = float(x[j])
        if val > 1e-7:
            recipe_runs[recipe.class_name] = val

    for cls, k in sink_idx.items():
        val = float(x[n_recipes + k])
        if val > 1e-7:
            sunk_items[cls] = val

    total_score = -float(res.fun)

    return SolveResult(
        total_score=total_score,
        recipe_runs=recipe_runs,
        sunk_items=sunk_items,
    )


def compute_leftovers(
    recipes: list[Recipe],
    recipe_runs: dict[str, float],
    base_supplies: dict[str, float],
    sunk_items: dict[str, float],
) -> dict[str, float]:
    """
    leftover[item] = base + produced - consumed - sunk
    Includes POWER_ITEM.
    """
    leftovers: dict[str, float] = dict(base_supplies)

    recipe_by_class = {r.class_name: r for r in recipes}

    for recipe_class, runs in recipe_runs.items():
        rec = recipe_by_class[recipe_class]

        for cls, amt in rec.ingredients:
            leftovers[cls] = leftovers.get(cls, 0.0) - amt * runs

        for cls, amt in rec.products:
            leftovers[cls] = leftovers.get(cls, 0.0) + amt * runs

        power_net_mj = recipe_power_net_mj(rec)
        if abs(power_net_mj) > EPS:
            leftovers[POWER_ITEM] = leftovers.get(POWER_ITEM, 0.0) + power_net_mj * runs

    for cls, amt in sunk_items.items():
        leftovers[cls] = leftovers.get(cls, 0.0) - amt

    return leftovers


def pretty_amount(x: float) -> str:
    if math.isclose(x, round(x), abs_tol=1e-9):
        return str(int(round(x)))
    return f"{x:.6f}".rstrip("0").rstrip(".")


def compute_recipe_energy_stats(
    recipes: list[Recipe],
    recipe_runs: dict[str, float],
) -> list[RecipeEnergyStat]:
    """
    Batch-energy estimate:
      energy_per_run_mj = abs(building_power_mw * duration_s)

    For display:
      - power_mw remains signed
      - energy_per_run_mj is always non-negative magnitude
      - total_energy_mj is always non-negative magnitude

    Sign of production/consumption is implied by power_mw:
      power_mw > 0 => consumer
      power_mw < 0 => producer
    """
    recipe_by_class = {r.class_name: r for r in recipes}
    stats: list[RecipeEnergyStat] = []

    for recipe_class, runs in recipe_runs.items():
        rec = recipe_by_class[recipe_class]
        building = pick_primary_building(rec)

        if building is None:
            continue

        power_mw = float(getattr(building, "power_consumption", 0.0) or 0.0)
        duration_s = float(getattr(rec, "duration_s", 0.0) or 0.0)

        energy_per_run_mj = abs(power_mw * duration_s)
        total_energy_mj = energy_per_run_mj * runs

        stats.append(
            RecipeEnergyStat(
                recipe_class=rec.class_name,
                recipe_name=getattr(rec, "name", rec.class_name),
                building_name=getattr(building, "name", getattr(building, "class_name", "Unknown")),
                building_class=getattr(building, "class_name", "Unknown"),
                runs=runs,
                power_mw=power_mw,
                duration_s=duration_s,
                energy_per_run_mj=energy_per_run_mj,
                total_energy_mj=total_energy_mj,
            )
        )

    return stats


def print_summary(
    items: list[Item],
    recipes: list[Recipe],
    result: SolveResult,
    base_supplies: dict[str, float],
    show_recipe_limit: int,
    show_sink_limit: int,
    show_leftover_limit: int,
    show_power_limit: int,
) -> None:
    items_by_class = {it.class_name: it for it in items}
    if POWER_ITEM not in items_by_class:
        items_by_class[POWER_ITEM] = Item(
            class_name=POWER_ITEM,
            name="Power (MJ)",
            sink_points=0,
        )

    print("=== OPTIMAL SOLUTION ===")
    print(f"Total sink score: {result.total_score:.3f}")
    print()

    if result.sunk_items:
        print(f"=== Sunk Items (top {show_sink_limit}) ===")
        sunk_sorted = sorted(
            result.sunk_items.items(),
            key=lambda kv: (
                -((items_by_class.get(kv[0]).sink_points if kv[0] in items_by_class else 0) * kv[1]),
                kv[0],
            ),
        )
        for cls, amt in sunk_sorted[:show_sink_limit]:
            item = items_by_class.get(cls)
            name = item.name if item else cls
            sink_points = item.sink_points if item else 0
            subtotal = sink_points * amt
            print(
                f"{name} ({cls}): amount={pretty_amount(amt)}, "
                f"points/item={sink_points}, subtotal={subtotal:.3f}"
            )
        print()
    else:
        print("No items were sunk.")
        print()

    if result.recipe_runs:
        print(f"=== Recipe Usage (top {show_recipe_limit}) ===")
        recipe_sorted = sorted(result.recipe_runs.items(), key=lambda kv: (-kv[1], kv[0]))
        recipe_by_class = {r.class_name: r for r in recipes}

        for recipe_class, runs in recipe_sorted[:show_recipe_limit]:
            rec = recipe_by_class[recipe_class]
            print(f"{getattr(rec, 'name', recipe_class)} ({recipe_class}): runs={pretty_amount(runs)}")
        print()
    else:
        print("No recipes were used.")
        print()

    energy_stats = compute_recipe_energy_stats(recipes, result.recipe_runs)

    if energy_stats:
        consuming_stats = [x for x in energy_stats if x.power_mw > EPS]
        producing_stats = [x for x in energy_stats if x.power_mw < -EPS]

        total_consumed_mj = sum(x.total_energy_mj for x in consuming_stats)
        total_produced_mj = sum(x.total_energy_mj for x in producing_stats)
        net_leftover_power_mj = compute_leftovers(
            recipes=recipes,
            recipe_runs=result.recipe_runs,
            base_supplies=base_supplies,
            sunk_items=result.sunk_items,
        ).get(POWER_ITEM, 0.0)

        total_consumer_nominal_power_mw = sum(x.power_mw for x in consuming_stats if x.runs > 1e-7)
        total_generator_nominal_power_mw = sum(-x.power_mw for x in producing_stats if x.runs > 1e-7)

        print("=== Power / Energy Summary ===")
        print(f"Recipes with resolved building power: {len(energy_stats)}")
        print(f"Total nominal consumer power:        {total_consumer_nominal_power_mw:.3f} MW")
        print(f"Total nominal generator power:       {total_generator_nominal_power_mw:.3f} MW")
        print(f"Total energy consumed:               {total_consumed_mj:.3f} MJ")
        print(f"Total energy produced:               {total_produced_mj:.3f} MJ")
        print(f"Net leftover energy:                 {net_leftover_power_mj:.3f} MJ")
        print(f"Net leftover energy:                 {net_leftover_power_mj / 3600.0:.6f} MWh")
        print()

        print(f"=== Recipe Energy Usage (top {show_power_limit}) ===")
        energy_sorted = sorted(energy_stats, key=lambda x: (-x.total_energy_mj, x.recipe_class))
        for s in energy_sorted[:show_power_limit]:
            role = "producer" if s.power_mw < 0 else "consumer"
            print(
                f"{s.recipe_name} ({s.recipe_class})\n"
                f"  building:        {s.building_name} ({s.building_class})\n"
                f"  role:            {role}\n"
                f"  runs:            {pretty_amount(s.runs)}\n"
                f"  power:           {s.power_mw:.3f} MW\n"
                f"  duration/run:    {s.duration_s:.3f} s\n"
                f"  energy/run:      {s.energy_per_run_mj:.3f} MJ\n"
                f"  total energy:    {s.total_energy_mj:.3f} MJ"
            )
        print()
    else:
        print("=== Power / Energy Summary ===")
        print("No recipe energy data could be derived from resolved building information.")
        print()

    leftovers = compute_leftovers(recipes, result.recipe_runs, base_supplies, result.sunk_items)
    leftovers = {k: v for k, v in leftovers.items() if v > 1e-7}

    if leftovers:
        print(f"=== Leftover Items (top {show_leftover_limit}) ===")
        leftover_sorted = sorted(leftovers.items(), key=lambda kv: (-kv[1], kv[0]))
        for cls, amt in leftover_sorted[:show_leftover_limit]:
            item = items_by_class.get(cls)
            name = item.name if item else cls

            if cls == POWER_ITEM:
                print(f"{name} ({cls}): leftover={amt:.3f} MJ ({amt / 3600.0:.6f} MWh)")
            else:
                print(f"{name} ({cls}): leftover={pretty_amount(amt)}")
        print()


def filter_valid_recipes_from_base(
    recipes: list[Recipe],
    base_supplies: dict[str, float],
    exclude_zero_input: bool = True,
) -> list[Recipe]:
    """
    Keep only recipes that are structurally reachable from the provided base items.

    A recipe is considered reachable if all its ingredients are already reachable.
    Reachability starts from the keys of base_supplies.

    This is a structural filter, not a quantitative one.

    POWER_ITEM is treated specially:
      - if a recipe consumes POWER_ITEM, that does NOT make it unreachable by itself,
        because power may be produced by other reachable recipes
      - if a recipe produces POWER_ITEM, it is judged by its non-power ingredients only
    """
    reachable_items = {cls for cls, amt in base_supplies.items() if amt > 0}
    valid_recipes: list[Recipe] = []
    remaining = list(recipes)

    changed = True
    while changed:
        changed = False
        next_remaining: list[Recipe] = []

        for rec in remaining:
            if exclude_zero_input and not rec.ingredients:
                next_remaining.append(rec)
                continue

            ingredient_classes = {cls for cls, _amt in rec.ingredients if cls != POWER_ITEM}

            if ingredient_classes.issubset(reachable_items):
                valid_recipes.append(rec)
                for cls, _amt in rec.products:
                    if cls not in reachable_items:
                        reachable_items.add(cls)
                        changed = True

                # If this recipe has a power contribution, mark POWER_ITEM as structurally reachable.
                power_net_mj = recipe_power_net_mj(rec)
                if abs(power_net_mj) > EPS and POWER_ITEM not in reachable_items:
                    reachable_items.add(POWER_ITEM)
                    changed = True

                changed = True
            else:
                next_remaining.append(rec)

        remaining = next_remaining

    return valid_recipes


def merge_supplies(defaults: dict[str, float], overrides: dict[str, float]) -> dict[str, float]:
    merged = dict(defaults)
    for k, v in overrides.items():
        merged[k] = merged.get(k, 0.0) + v
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Maximize Satisfactory sink score from base ingredients using LP, including power as a produced/consumed resource."
    )
    parser.add_argument(
        "json_path",
        help="Path to the game JSON file consumed by load_data.py",
    )
    parser.add_argument(
        "supplies",
        nargs="*",
        help='Extra base supplies like Desc_OreIron_C=10000 Desc_OreCopper_C=5000 __POWER_MJ__=100000',
    )
    parser.add_argument(
        "--top-recipes",
        type=int,
        default=10,
        help="How many recipe usages to print",
    )
    parser.add_argument(
        "--top-sinks",
        type=int,
        default=10,
        help="How many sunk items to print",
    )
    parser.add_argument(
        "--top-leftovers",
        type=int,
        default=10,
        help="How many leftovers to print",
    )
    parser.add_argument(
        "--top-power",
        type=int,
        default=10,
        help="How many recipe energy rows to print",
    )
    parser.add_argument(
        "--ignore-default-supplies",
        action="store_true",
        help="Ignore the built-in base supply preset and only use supplies passed on the CLI",
    )

    args = parser.parse_args()

    cli_supplies = parse_supply_args(args.supplies)

    default_base_supplies = {
        "Desc_OreIron_C": 92100,
        "Desc_Water_C": 100000000,
        "Desc_OreCopper_C": 36900,
        "Desc_LiquidOil_C": 12600,
        "Desc_NitrogenGas_C": 12000,
        "Desc_Coal_C": 42300,
        "Desc_Stone_C": 69300,
        "Desc_OreGold_C": 15000,
        "Desc_RawQuartz_C": 13500,
        "Desc_Sulfur_C": 10800,
        "Desc_OreBauxite_C": 12300,
        "Desc_SAM_C": 10200,
        "Desc_OreUranium_C": 2100,
        POWER_ITEM: 0.0,
    }

    if args.ignore_default_supplies:
        base_supplies = cli_supplies
        base_supplies.setdefault(POWER_ITEM, 0.0)
    else:
        base_supplies = merge_supplies(default_base_supplies, cli_supplies)

    # load_model now returns 4 values
    items, recipes, _recipes_by_product, _buildings_by_class = load_model(args.json_path)

    filtered_recipes = filter_valid_recipes_from_base(
        recipes,
        base_supplies=base_supplies,
        exclude_zero_input=False,
    )

    print(f"Loaded items:            {len(items)}")
    print(f"Loaded recipes:          {len(recipes)}")
    print(f"Reachable valid recipes: {len(filtered_recipes)}")
    print(f"Base supplies:           {len(base_supplies)} item types")
    print(f"Power item key:          {POWER_ITEM}")
    print()

    if not base_supplies:
        print("Warning: no base supplies were given. The optimal score will likely be 0.")
        print()

    result = solve_max_sink_score(
        items=items,
        recipes=filtered_recipes,
        base_supplies=base_supplies,
    )

    print_summary(
        items=items,
        recipes=filtered_recipes,
        result=result,
        base_supplies=base_supplies,
        show_recipe_limit=args.top_recipes,
        show_sink_limit=args.top_sinks,
        show_leftover_limit=args.top_leftovers,
        show_power_limit=args.top_power,
    )


if __name__ == "__main__":
    main()