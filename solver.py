from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import json

from scipy.optimize import linprog

from load_data import load_model, Item, Recipe


EPS = 1e-9

# Synthetic resource keys
POWER_ITEM = "__POWER_MW__"
POWERSHARD_ITEM = "__POWERSHARD__"
SOMERSLOOP_ITEM = "__SOMERSLOOP__"

# Leftovers for these items are not allowed.
# The optimizer must ensure their final balance is exactly zero.
FORBIDDEN_LEFTOVERS: set[str] = {
    "Desc_PlutoniumWaste_C",
    "Desc_NuclearWaste_C",
    "Desc_PlutoniumPellet_C",
    "Desc_PlutoniumCell_C",
    "Desc_NonFissibleUranium_C"
}

# Satisfactory production-building overclock exponent
PRODUCTION_POWER_EXPONENT = 1.3219280948873624

# Discrete clock options to keep the optimization linear
CLOCK_OPTIONS = (
    1.00,
    1.25,
    1.50,
    1.75,
    2.00,
    2.25,
    2.50,
)

# Heuristic/default Somersloop slot counts for full 2.0 amplification.
# You can extend this mapping if your data contains more specific building classes.
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
AUGMENT_SOMERSLOOP_COST = 13

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
    power_net_mw: float  # + produces power, - consumes power


@dataclass(frozen=True, slots=True)
class SolveResult:
    total_score: float
    augment_count: int
    mode_usage: dict[str, float]         # mode_id -> building count
    sunk_items: dict[str, float]         # item -> amount/min
    leftover_items: dict[str, float]     # item -> amount/min
    total_power_produced_mw: float
    total_power_consumed_mw: float
    net_power_left_mw: float


def parse_supply_args(values: list[str]) -> dict[str, float]:
    """
    Parse CLI values like:
        Desc_OreIron_C=10000
        __POWER_MW__=500
        __POWERSHARD__=30
        __SOMERSLOOP__=12
    """
    supplies: dict[str, float] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Supply must be ITEM=AMOUNT, got: {raw!r}")

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


def merge_supplies(defaults: dict[str, float], overrides: dict[str, float]) -> dict[str, float]:
    merged = dict(defaults)
    for k, v in overrides.items():
        merged[k] = merged.get(k, 0.0) + v
    return merged


def pretty_amount(x: float) -> str:
    if math.isclose(x, round(x), abs_tol=1e-9):
        return str(int(round(x)))
    return f"{x:.6f}".rstrip("0").rstrip(".")


def pick_primary_building(recipe: Recipe):
    """
    Prefer real buildables over helper/workbench components.
    """
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
    """
    Ingredient consumption per minute at 100% clock and no Somersloops.
    Positive numbers mean "consumes this much per minute".
    """
    duration_min = recipe_duration_minutes(recipe)
    if duration_min <= 0:
        return {}

    rates: dict[str, float] = {}
    for cls, amt in recipe.ingredients:
        rates[cls] = rates.get(cls, 0.0) + float(amt) / duration_min
    return rates


def recipe_base_product_rates_per_min(recipe: Recipe) -> dict[str, float]:
    """
    Product output per minute at 100% clock and no Somersloops.
    Positive numbers mean "produces this much per minute".
    """
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
    """
    True for miners / extractors that generate raw materials from a fixed source.

    For these buildings, overclocking should increase only product output,
    while the "ingredient" that represents source occupancy (node / patch / well)
    should stay constant.
    """
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

    # Fallback for data sets with slightly different names/classes
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
    """
    100%     -> 0 shards
    >100-150 -> 1 shard
    >150-200 -> 2 shards
    >200-250 -> 3 shards
    """
    if clock <= 1.0 + EPS:
        return 0
    if clock <= 1.5 + EPS:
        return 1
    if clock <= 2.0 + EPS:
        return 2
    return 3


def somersloop_output_multiplier(sloops_used: int, max_slots: int) -> float:
    """
    0/max -> 1.0x
    full  -> 2.0x

    Linear interpolation across slots.
    """
    if max_slots <= 0:
        return 1.0
    return 1.0 + (float(sloops_used) / float(max_slots))


def somersloop_power_multiplier(sloops_used: int, max_slots: int) -> float:
    """
    Chosen so:
      0 Somersloops -> 1x power
      full boost    -> 4x power
    """
    out_mult = somersloop_output_multiplier(sloops_used, max_slots)
    return out_mult * out_mult


def effective_shard_slot_size(recipe: Recipe) -> int:
    """
    Raw-resource producers should always be allowed to use up to 3 Power Shards,
    even if the imported data reports fewer shard slots.
    """
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
    """
    Returns net power contribution per building:
      positive => power produced
      negative => power consumed
    """
    building = pick_primary_building(recipe)
    if building is None:
        return 0.0

    base_power = float(getattr(building, "power_consumption", 0.0) or 0.0)

    if base_power < 0.0:
        # Generator / power producer:
        # use linear clock scaling
        return (-base_power) * clock

    # Production buildings consume more power with clock speed and Somersloops
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

                # Ingredients:
                # - normal buildings: scale with clock
                # - raw resource generators: their source occupancy stays constant
                for cls, amt_per_min in base_ing_rates.items():
                    if raw_generator:
                        scaled = -amt_per_min
                    else:
                        scaled = -amt_per_min * clock

                    if abs(scaled) > EPS:
                        net[cls] = net.get(cls, 0.0) + scaled

                # Products scale with clock and Somersloop output multiplier
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


def build_item_lookup(items: list[Item], recipes: list[Recipe], base_supplies: dict[str, float]) -> dict[str, Item]:
    items_by_class: dict[str, Item] = {it.class_name: it for it in items}

    all_item_classes = set(items_by_class.keys())
    all_item_classes.update(base_supplies.keys())
    all_item_classes.update({POWER_ITEM, POWERSHARD_ITEM, SOMERSLOOP_ITEM})

    for rec in recipes:
        for cls, _ in rec.ingredients:
            all_item_classes.add(cls)
        for cls, _ in rec.products:
            all_item_classes.add(cls)

    for cls in all_item_classes:
        if cls not in items_by_class:
            pretty_name = {
                POWER_ITEM: "Power (MW)",
                POWERSHARD_ITEM: "Power Shards",
                SOMERSLOOP_ITEM: "Somersloops",
            }.get(cls, cls)
            items_by_class[cls] = Item(class_name=cls, name=pretty_name, sink_points=0)

    return items_by_class


def augment_fixed_nets_per_min(augment_recipe: Recipe | None, augment_count: int) -> tuple[dict[str, float], float]:
    """
    Returns:
      - fixed item/power net from running `augment_count` fueled augmenters
      - additive power grid multiplier contribution

    Example:
      one augmenter with power_grid_multiplier=1.3 contributes +0.3
      to the global power generation multiplier.
    """
    if augment_recipe is None or augment_count <= 0:
        return {}, 0.0

    building = pick_primary_building(augment_recipe)
    if building is None:
        return {}, 0.0

    duration_min = recipe_duration_minutes(augment_recipe)
    if duration_min <= 0:
        return {}, 0.0

    runs_per_min_per_building = 1.0 / duration_min
    total_runs_per_min = augment_count * runs_per_min_per_building

    net: dict[str, float] = {}

    for cls, amt in augment_recipe.products:
        net[cls] = net.get(cls, 0.0) + float(amt) * total_runs_per_min

    for cls, amt in augment_recipe.ingredients:
        net[cls] = net.get(cls, 0.0) - float(amt) * total_runs_per_min

    # The augmenter itself can also generate direct power via negative power consumption
    base_power = float(getattr(building, "power_consumption", 0.0) or 0.0)
    if base_power < 0.0:
        net[POWER_ITEM] = net.get(POWER_ITEM, 0.0) + (-base_power) * augment_count

    grid_mult = float(getattr(building, "power_grid_multiplier", 1.0) or 1.0)
    additive_boost = max(0.0, grid_mult - 1.0) * augment_count

    return net, additive_boost


def compute_leftovers(
    modes: list[RecipeMode],
    mode_usage: dict[str, float],
    base_supplies: dict[str, float],
    sunk_items: dict[str, float],
    fixed_nets: dict[str, float],
    power_boost_multiplier: float,
    augment_count: int,
) -> dict[str, float]:
    leftovers: dict[str, float] = dict(base_supplies)

    for cls, amt in fixed_nets.items():
        leftovers[cls] = leftovers.get(cls, 0.0) + amt

    mode_by_id = {m.mode_id: m for m in modes}

    for mode_id, count in mode_usage.items():
        mode = mode_by_id[mode_id]

        for cls, amt in mode.item_net_per_min.items():
            leftovers[cls] = leftovers.get(cls, 0.0) + amt * count

        if mode.power_net_mw > EPS:
            leftovers[POWER_ITEM] = leftovers.get(POWER_ITEM, 0.0) + mode.power_net_mw * power_boost_multiplier * count
        elif mode.power_net_mw < -EPS:
            leftovers[POWER_ITEM] = leftovers.get(POWER_ITEM, 0.0) + mode.power_net_mw * count

        leftovers[POWERSHARD_ITEM] = leftovers.get(POWERSHARD_ITEM, 0.0) - float(mode.power_shards) * count
        leftovers[SOMERSLOOP_ITEM] = leftovers.get(SOMERSLOOP_ITEM, 0.0) - float(mode.somersloops) * count

    if augment_count > 0:
        leftovers[SOMERSLOOP_ITEM] = leftovers.get(SOMERSLOOP_ITEM, 0.0) - AUGMENT_SOMERSLOOP_COST * float(augment_count)

    for cls, amt in sunk_items.items():
        leftovers[cls] = leftovers.get(cls, 0.0) - amt

    cleaned: dict[str, float] = {}
    for k, v in leftovers.items():
        if abs(v) <= 1e-7:
            v = 0.0
        if v > 1e-7 or k in {POWER_ITEM, POWERSHARD_ITEM, SOMERSLOOP_ITEM} or k in FORBIDDEN_LEFTOVERS:
            cleaned[k] = v

    return cleaned


def solve_for_fixed_augment_count(
    items: list[Item],
    recipes: list[Recipe],
    base_supplies: dict[str, float],
    modes: list[RecipeMode],
    augment_recipe: Recipe | None,
    augment_count: int,
    forbidden_leftovers: set[str] | None = None,
) -> SolveResult | None:
    forbidden_leftovers = forbidden_leftovers or set()

    if AUGMENT_SOMERSLOOP_COST  * augment_count > float(base_supplies.get(SOMERSLOOP_ITEM, 0.0)) + EPS:
        return None

    items_by_class = build_item_lookup(items, recipes, base_supplies)
    item_classes = sorted(items_by_class.keys())

    fixed_nets, additive_boost = augment_fixed_nets_per_min(augment_recipe, augment_count)
    power_boost_multiplier = 1.0 + additive_boost

    sinkable_items = [cls for cls in item_classes if items_by_class[cls].sink_points > 0]
    sink_idx = {cls: k for k, cls in enumerate(sinkable_items)}

    n_modes = len(modes)
    n_sink = len(sinkable_items)
    n_vars = n_modes + n_sink

    # scipy.linprog minimizes
    c = [0.0] * n_vars
    for cls, k in sink_idx.items():
        c[n_modes + k] = -float(items_by_class[cls].sink_points)

    A_ub: list[list[float]] = []
    b_ub: list[float] = []

    for cls in item_classes:
        row = [0.0] * n_vars

        for j, mode in enumerate(modes):
            amt = mode.item_net_per_min.get(cls, 0.0)

            if cls == POWER_ITEM:
                amt = 0.0
                if mode.power_net_mw > EPS:
                    amt = mode.power_net_mw * power_boost_multiplier
                elif mode.power_net_mw < -EPS:
                    amt = mode.power_net_mw

            if abs(amt) > EPS:
                row[j] = -amt

        sk = sink_idx.get(cls)
        if sk is not None:
            row[n_modes + sk] = 1.0

        effective_supply = float(base_supplies.get(cls, 0.0)) + float(fixed_nets.get(cls, 0.0))

        if cls in forbidden_leftovers:
            # Enforce:
            #   effective_supply + net_production - sunk == 0
            #
            # Current row already gives:
            #   -net_production + sunk <= effective_supply
            #
            # Add the opposite inequality too, forcing equality.
            A_ub.append(row)
            b_ub.append(effective_supply)

            A_ub.append([-v for v in row])
            b_ub.append(-effective_supply)
        else:
            A_ub.append(row)
            b_ub.append(effective_supply)

    # Global Power Shard budget
    row_ps = [0.0] * n_vars
    for j, mode in enumerate(modes):
        if mode.power_shards > 0:
            row_ps[j] = float(mode.power_shards)
    A_ub.append(row_ps)
    b_ub.append(float(base_supplies.get(POWERSHARD_ITEM, 0.0)))

    # Global Somersloop budget
    row_sl = [0.0] * n_vars
    for j, mode in enumerate(modes):
        if mode.somersloops > 0:
            row_sl[j] = float(mode.somersloops)
    A_ub.append(row_sl)
    b_ub.append(
        float(base_supplies.get(SOMERSLOOP_ITEM, 0.0))
        - float(AUGMENT_SOMERSLOOP_COST) * float(augment_count)
    )

    bounds = [(0.0, None)] * n_vars

    res = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        return None

    x = res.x

    mode_usage: dict[str, float] = {}
    sunk_items: dict[str, float] = {}

    for j, mode in enumerate(modes):
        val = float(x[j])
        if val > 1e-7:
            mode_usage[mode.mode_id] = val

    for cls, k in sink_idx.items():
        val = float(x[n_modes + k])
        if val > 1e-7:
            sunk_items[cls] = val

    leftovers = compute_leftovers(
        modes=modes,
        mode_usage=mode_usage,
        base_supplies=base_supplies,
        sunk_items=sunk_items,
        fixed_nets=fixed_nets,
        power_boost_multiplier=power_boost_multiplier,
        augment_count=augment_count,
    )

    mode_by_id = {m.mode_id: m for m in modes}

    total_power_produced_mw = 0.0
    total_power_consumed_mw = 0.0

    for mode_id, count in mode_usage.items():
        mode = mode_by_id[mode_id]

        if mode.power_net_mw > EPS:
            total_power_produced_mw += mode.power_net_mw * power_boost_multiplier * count
        elif mode.power_net_mw < -EPS:
            total_power_consumed_mw += (-mode.power_net_mw) * count

    fixed_power = fixed_nets.get(POWER_ITEM, 0.0)
    if fixed_power > EPS:
        total_power_produced_mw += fixed_power * power_boost_multiplier

    total_score = -float(res.fun)

    return SolveResult(
        total_score=total_score,
        augment_count=augment_count,
        mode_usage=mode_usage,
        sunk_items=sunk_items,
        leftover_items=leftovers,
        total_power_produced_mw=total_power_produced_mw,
        total_power_consumed_mw=total_power_consumed_mw,
        net_power_left_mw=leftovers.get(POWER_ITEM, 0.0),
    )


def solve_max_sink_score(
    items: list[Item],
    recipes: list[Recipe],
    base_supplies: dict[str, float],
    forbidden_leftovers: set[str] | None = None,
) -> SolveResult:
    forbidden_leftovers = forbidden_leftovers or set()

    modes, augment_recipe = build_recipe_modes(recipes)

    max_sloops = int(math.floor(base_supplies.get(SOMERSLOOP_ITEM, 0.0)))
    max_augments_by_sloops = max_sloops // AUGMENT_SOMERSLOOP_COST  if augment_recipe is not None else 0

    best: SolveResult | None = None

    for augment_count in range(max_augments_by_sloops + 1):
        result = solve_for_fixed_augment_count(
            items=items,
            recipes=recipes,
            base_supplies=base_supplies,
            modes=modes,
            augment_recipe=augment_recipe,
            augment_count=augment_count,
            forbidden_leftovers=forbidden_leftovers,
        )
        if result is None:
            continue

        if best is None or result.total_score > best.total_score + 1e-7:
            best = result

    if best is None:
        raise RuntimeError("Optimization failed for every tested augment count.")

    return best


def filter_valid_recipes_from_base(
    recipes: list[Recipe],
    base_supplies: dict[str, float],
    exclude_zero_input: bool = False,
) -> list[Recipe]:
    """
    Structural reachability filter.

    POWER / POWERSHARDS / SOMERSLOOPS do not block reachability.
    """
    special = {POWER_ITEM, POWERSHARD_ITEM, SOMERSLOOP_ITEM}
    reachable_items = {cls for cls, amt in base_supplies.items() if amt > 0 and cls not in special}

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

            ingredient_classes = {cls for cls, _ in rec.ingredients if cls not in special}

            if ingredient_classes.issubset(reachable_items):
                valid_recipes.append(rec)
                for cls, _ in rec.products:
                    if cls not in reachable_items:
                        reachable_items.add(cls)
                        changed = True
            else:
                next_remaining.append(rec)

        remaining = next_remaining

    return valid_recipes


def print_summary(
    items: list[Item],
    recipes: list[Recipe],
    result: SolveResult,
    top_modes: int,
    top_sinks: int,
    top_leftovers: int,
) -> None:
    items_by_class = {it.class_name: it for it in items}
    items_by_class.setdefault(POWER_ITEM, Item(POWER_ITEM, "Power (MW)", 0))
    items_by_class.setdefault(POWERSHARD_ITEM, Item(POWERSHARD_ITEM, "Power Shards", 0))
    items_by_class.setdefault(SOMERSLOOP_ITEM, Item(SOMERSLOOP_ITEM, "Somersloops", 0))

    modes, _augment_recipe = build_recipe_modes(recipes)
    mode_by_id = {m.mode_id: m for m in modes}

    print("=== OPTIMAL SOLUTION ===")
    print(f"Total sink score: {result.total_score:.3f} / min")
    print(f"Chosen fueled augmenters: {result.augment_count}")
    print()

    print("=== Power Summary ===")
    print(f"Total power produced:  {result.total_power_produced_mw:.3f} MW")
    print(f"Total power consumed:  {result.total_power_consumed_mw:.3f} MW")
    print(f"Net power leftover:    {result.net_power_left_mw:.3f} MW")
    print()

    if result.sunk_items:
        print(f"=== Sunk Items (top {top_sinks}) ===")
        sunk_sorted = sorted(
            result.sunk_items.items(),
            key=lambda kv: (
                -((items_by_class[kv[0]].sink_points if kv[0] in items_by_class else 0) * kv[1]),
                kv[0],
            ),
        )
        for cls, amt in sunk_sorted[:top_sinks]:
            item = items_by_class.get(cls)
            name = item.name if item else cls
            sink_points = item.sink_points if item else 0
            subtotal = sink_points * amt
            print(
                f"{name} ({cls}): amount/min={pretty_amount(amt)}, "
                f"points/item={sink_points}, subtotal/min={subtotal:.3f}"
            )
        print()
    else:
        print("No items were sunk.")
        print()

    if result.mode_usage:
        print(f"=== Recipe / Mode Usage (top {top_modes}) ===")
        mode_sorted = sorted(result.mode_usage.items(), key=lambda kv: (-kv[1], kv[0]))
        for mode_id, buildings in mode_sorted[:top_modes]:
            m = mode_by_id[mode_id]
            print(
                f"{m.recipe_name} ({m.recipe_class})\n"
                f"  building:       {m.building_name} ({m.building_class})\n"
                f"  buildings:      {pretty_amount(buildings)}\n"
                f"  clock:          {m.clock:.2f}x\n"
                f"  power shards:   {m.power_shards} per building\n"
                f"  somersloops:    {m.somersloops}/{m.somersloop_slots} per building\n"
                f"  out multiplier: {m.output_multiplier:.3f}x\n"
                f"  pwr multiplier: {m.power_multiplier:.3f}x\n"
                f"  net power:      {m.power_net_mw:.3f} MW/building"
            )
        print()
    else:
        print("No recipe modes were used.")
        print()

    if result.leftover_items:
        print(f"=== Leftovers (top {top_leftovers}) ===")
        leftover_sorted = sorted(
            result.leftover_items.items(),
            key=lambda kv: (-(kv[1] if kv[1] > 0 else -1e18), kv[0]),
        )
        for cls, amt in leftover_sorted[:top_leftovers]:
            item = items_by_class.get(cls)
            name = item.name if item else cls

            if cls == POWER_ITEM:
                print(f"{name} ({cls}): {amt:.3f} MW")
            else:
                print(f"{name} ({cls}): {pretty_amount(amt)} / min")
        print()

    print("=== Special Resource Leftovers ===")
    for cls in [POWERSHARD_ITEM, SOMERSLOOP_ITEM, POWER_ITEM]:
        amt = result.leftover_items.get(cls, 0.0)
        item = items_by_class.get(cls)
        name = item.name if item else cls

        if cls == POWER_ITEM:
            print(f"{name} ({cls}): {amt:.3f} MW")
        else:
            print(f"{name} ({cls}): {pretty_amount(amt)}")
    print()


def solve_result_to_dict(result: SolveResult) -> dict:
    return {
        "total_score": result.total_score,
        "augment_count": result.augment_count,
        "mode_usage": result.mode_usage,
        "sunk_items": result.sunk_items,
        "leftover_items": result.leftover_items,
        "total_power_produced_mw": result.total_power_produced_mw,
        "total_power_consumed_mw": result.total_power_consumed_mw,
        "net_power_left_mw": result.net_power_left_mw,
    }


def save_result_json(path: str | Path, result: SolveResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(solve_result_to_dict(result), f, indent=2, sort_keys=True)

    print(f"Saved result to: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Maximize Satisfactory sink score from per-minute base supplies, "
            "including Power Shards, Somersloops, fueled Alien Power Augmenters, "
            "overclocking, and Somersloop amplification."
        )
    )
    parser.add_argument(
        "json_path",
        help="Path to the game JSON file consumed by load_data.py",
    )
    parser.add_argument(
        "supplies",
        nargs="*",
        help=(
            "Extra base supplies like "
            "Desc_OreIron_C=10000 Desc_OreCopper_C=5000 "
            "__POWER_MW__=1000 __POWERSHARD__=120 __SOMERSLOOP__=30"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to save the optimal result as JSON",
    )
    parser.add_argument("--top-modes", type=int, default=10)
    parser.add_argument("--top-sinks", type=int, default=10)
    parser.add_argument("--top-leftovers", type=int, default=10)
    parser.add_argument(
        "--ignore-default-supplies",
        action="store_true",
        help="Ignore the built-in base supply preset and only use supplies passed on the CLI",
    )

    args = parser.parse_args()
    cli_supplies = parse_supply_args(args.supplies)

    default_base_supplies = {
        # Iron
        "IronOre_PatchImpure_C": 39,
        "IronOre_PatchNormal_C": 42,
        "IronOre_PatchPure_C": 46,

        # Copper
        "CopperOre_PatchImpure_C": 13,
        "CopperOre_PatchNormal_C": 29,
        "CopperOre_PatchPure_C": 13,

        # Limestone
        "Limestone_PatchImpure_C": 15,
        "Limestone_PatchNormal_C": 50,
        "Limestone_PatchPure_C": 29,

        # Coal
        "Coal_PatchImpure_C": 15,
        "Coal_PatchNormal_C": 31,
        "Coal_PatchPure_C": 16,

        # Caterium
        "CateriumOre_PatchImpure_C": 0,
        "CateriumOre_PatchNormal_C": 9,
        "CateriumOre_PatchPure_C": 8,

        # Raw Quartz
        "RawQuartz_PatchImpure_C": 3,
        "RawQuartz_PatchNormal_C": 7,
        "RawQuartz_PatchPure_C": 7,

        # Sulfur
        "Sulfur_PatchImpure_C": 6,
        "Sulfur_PatchNormal_C": 5,
        "Sulfur_PatchPure_C": 5,

        # Bauxite
        "Bauxite_PatchImpure_C": 5,
        "Bauxite_PatchNormal_C": 6,
        "Bauxite_PatchPure_C": 6,

        # Uranium
        "Uranium_PatchImpure_C": 3,
        "Uranium_PatchNormal_C": 2,
        "Uranium_PatchPure_C": 0,

        # SAM
        "SAM_PatchImpure_C": 10,
        "SAM_PatchNormal_C": 6,
        "SAM_PatchPure_C": 3,

        # Crude Oil nodes
        "CrudeOil_NodeImpure_C": 10,
        "CrudeOil_NodeNormal_C": 12,
        "CrudeOil_NodePure_C": 8,

        # Crude Oil wells (satellite nodes)
        "CrudeOil_WellImpure_C": 8,
        "CrudeOil_WellNormal_C": 6,
        "CrudeOil_WellPure_C": 4,

        # Water wells (satellite nodes)
        #"Water_WellImpure_C": 7,
        #"Water_WellNormal_C": 12,
        #"Water_WellPure_C": 36,

        # Nitrogen Gas wells (satellite nodes)
        "NitrogenGas_WellImpure_C": 2,
        "NitrogenGas_WellNormal_C": 7,
        "NitrogenGas_WellPure_C": 36,

        POWER_ITEM: 7970.0,
        POWERSHARD_ITEM: 2651,
        SOMERSLOOP_ITEM: 104,
    }

    if args.ignore_default_supplies:
        base_supplies = cli_supplies
        base_supplies.setdefault(POWER_ITEM, 0.0)
        base_supplies.setdefault(POWERSHARD_ITEM, 0.0)
        base_supplies.setdefault(SOMERSLOOP_ITEM, 0.0)
    else:
        base_supplies = merge_supplies(default_base_supplies, cli_supplies)

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
    print(f"Power key:               {POWER_ITEM}")
    print(f"Power Shard key:         {POWERSHARD_ITEM}")
    print(f"Somersloop key:          {SOMERSLOOP_ITEM}")
    if FORBIDDEN_LEFTOVERS:
        print(f"Forbidden leftovers:     {', '.join(sorted(FORBIDDEN_LEFTOVERS))}")
    print()

    result = solve_max_sink_score(
        items=items,
        recipes=filtered_recipes,
        base_supplies=base_supplies,
        forbidden_leftovers=FORBIDDEN_LEFTOVERS,
    )
    if args.output_json:
        save_result_json(args.output_json, result)

    print_summary(
        items=items,
        recipes=filtered_recipes,
        result=result,
        top_modes=args.top_modes,
        top_sinks=args.top_sinks,
        top_leftovers=args.top_leftovers,
    )


if __name__ == "__main__":
    main()