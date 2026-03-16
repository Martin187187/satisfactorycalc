from __future__ import annotations

from dataclasses import dataclass
import re
import orjson  # pip install orjson
from scipy.optimize import linprog  # pip install scipy


PAIR_RE = re.compile(
    r'ItemClass="[^"]*\.([A-Za-z0-9_]+_C)\'",Amount=([0-9]+)'
)

# Matches things like:
# "/Game/FactoryGame/Buildable/Factory/ManufacturerMk1/Build_ManufacturerMk1.Build_ManufacturerMk1_C"
# and extracts Build_ManufacturerMk1_C
CLASS_REF_RE = re.compile(r'([A-Za-z0-9_]+_C)"?')

RECIPE_NATIVE_CLASS = "/Script/CoreUObject.Class'/Script/FactoryGame.FGRecipe'"
ITEM_NATIVE_CLASS = "/Script/CoreUObject.Class'/Script/FactoryGame.FGItemDescriptor'"

SINK_POINT_OVERRIDES = {
    "Desc_DarkEnergy_C": 0,
    "Desc_QuantumEnergy_C": 0,
    "Desc_Water_C": 0,
    "Desc_NitrogenGas_C": 0,
    "Desc_LiquidOil_C": 0,
    "Desc_HeavyOilResidue_C": 0,
    "Desc_LiquidFuel_C": 0,
    "Desc_LiquidBiofuel_C": 0,
    "Desc_Turbofuel_C": 0,
    "Desc_AluminaSolution_C": 0,
    "Desc_SulfuricAcid_C": 0,
    "Desc_NitricAcid_C": 0,
    "Desc_RocketFuel_C": 0,
    "Desc_IonizedFuel_C": 0,
    "Desc_DissolvedSilica_C": 0,
    "Desc_LiquidTurboFuel_C": 0,
}

AMOUNT_DIVISORS = {
    "Desc_Water_C": 1000.0,
    "Desc_NitrogenGas_C": 1000.0,
    "Desc_LiquidOil_C": 1000.0,
    "Desc_HeavyOilResidue_C": 1000.0,
    "Desc_LiquidFuel_C": 1000.0,
    "Desc_LiquidBiofuel_C": 1000.0,
    "Desc_Turbofuel_C": 1000.0,
    "Desc_AluminaSolution_C": 1000.0,
    "Desc_SulfuricAcid_C": 1000.0,
    "Desc_NitricAcid_C": 1000.0,
    "Desc_RocketFuel_C": 1000.0,
    "Desc_IonizedFuel_C": 1000.0,
    "Desc_DissolvedSilica_C": 1000.0,
    "Desc_DarkEnergy_C": 1000.0,
    "Desc_QuantumEnergy_C": 1000.0,
    "Desc_LiquidTurboFuel_C": 1000.0,
}


@dataclass(frozen=True, slots=True)
class Item:
    class_name: str
    name: str
    sink_points: int


@dataclass(frozen=True, slots=True)
class Building:
    class_name: str
    name: str
    power_consumption: float
    power_consumption_exponent: float
    production_shard_slot_size: int
    production_shard_boost_multiplier: float


@dataclass(frozen=True, slots=True)
class Recipe:
    class_name: str
    name: str
    duration_s: float
    ingredients: tuple[tuple[str, float], ...]   # (ClassName, amount)
    products: tuple[tuple[str, float], ...]
    produced_in: tuple[str, ...]                 # building class names only
    produced_in_buildings: tuple[Building, ...]  # resolved building data


@dataclass(frozen=True, slots=True)
class SuspiciousCycleResult:
    has_profitable_cycle: bool
    objective_value: float
    recipe_usage: dict[str, float]
    sink_output: dict[str, float]


def scale_amount(item_class: str, amount: float) -> float:
    divisor = AMOUNT_DIVISORS.get(item_class, 1.0)
    return amount / divisor


def parse_item_list(s: str) -> tuple[tuple[str, float], ...]:
    pairs: list[tuple[str, float]] = []

    for m in PAIR_RE.finditer(s or ""):
        item_class = m.group(1)
        raw_amount = float(m.group(2))
        amount = scale_amount(item_class, raw_amount)
        pairs.append((item_class, amount))

    return tuple(pairs)


def parse_class_refs(s: str) -> tuple[str, ...]:
    """
    Parse strings like:
      ("/Game/.../Build_ManufacturerMk1.Build_ManufacturerMk1_C",
       "/Game/.../BP_WorkshopComponent.BP_WorkshopComponent_C")

    and return:
      ("Build_ManufacturerMk1_C", "BP_WorkshopComponent_C")
    """
    if not s:
        return ()

    found = CLASS_REF_RE.findall(s)
    # preserve order, remove duplicates
    seen = set()
    result: list[str] = []
    for cls in found:
        if cls not in seen:
            seen.add(cls)
            result.append(cls)
    return tuple(result)


def to_float(value, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def load_json_any_encoding(path: str):
    with open(path, "rb") as f:
        raw = f.read()

    if raw.startswith(b"\xef\xbb\xbf"):
        text = raw.decode("utf-8-sig")
    elif raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        if b"\x00" in raw[:200]:
            text = raw.decode("utf-16")
        else:
            text = raw.decode("utf-8")

    return orjson.loads(text)


def extract_all_classes(data: list[dict]) -> dict[str, dict]:
    """
    Build a lookup of every class record by ClassName, regardless of NativeClass.
    This is the important step that lets recipes resolve buildings from other sections.
    """
    all_classes_by_name: dict[str, dict] = {}

    for group in data:
        if not isinstance(group, dict):
            continue

        classes = group.get("Classes", [])
        if not isinstance(classes, list):
            continue

        for cls in classes:
            if not isinstance(cls, dict):
                continue

            class_name = cls.get("ClassName", "")
            if not class_name:
                continue

            all_classes_by_name[class_name] = cls

    return all_classes_by_name


def build_building_lookup(all_classes_by_name: dict[str, dict]) -> dict[str, Building]:
    """
    Convert every class record that looks like a production-capable buildable/workbench
    into a Building. We do not depend on a single NativeClass here.
    """
    buildings: dict[str, Building] = {}

    for class_name, raw in all_classes_by_name.items():
        # Only keep classes that appear to be usable crafting buildings/components.
        # This heuristic is broad on purpose so workshop/workbench components are also picked up.
        has_relevant_fields = any(
            key in raw
            for key in (
                "mPowerConsumption",
                "mPowerConsumptionExponent",
                "mProductionShardSlotSize",
                "mProductionShardBoostMultiplier",
                "mManufacturingSpeed",
            )
        )

        if not has_relevant_fields:
            continue

        buildings[class_name] = Building(
            class_name=class_name,
            name=raw.get("mDisplayName", class_name),
            power_consumption=to_float(raw.get("mPowerConsumption", "0")),
            power_consumption_exponent=to_float(raw.get("mPowerConsumptionExponent", "0")),
            production_shard_slot_size=to_int(raw.get("mProductionShardSlotSize", "0")),
            production_shard_boost_multiplier=to_float(raw.get("mProductionShardBoostMultiplier", "0")),
        )

    return buildings


def load_model(path: str):
    data = load_json_any_encoding(path)

    if not isinstance(data, list):
        raise TypeError(f"Expected top-level JSON array, got {type(data).__name__}")

    all_classes_by_name = extract_all_classes(data)
    buildings_by_class = build_building_lookup(all_classes_by_name)

    items: list[Item] = []
    recipes: list[Recipe] = []

    for group in data:
        if not isinstance(group, dict):
            continue

        native_class = group.get("NativeClass", "")
        classes = group.get("Classes", [])
        if not isinstance(classes, list):
            continue

        for r in classes:
            if not isinstance(r, dict):
                continue

            cn = r.get("ClassName", "")
            if not cn:
                continue

            if native_class == ITEM_NATIVE_CLASS:
                sink_points = to_int(r.get("mResourceSinkPoints", "0"))
                sink_points = SINK_POINT_OVERRIDES.get(cn, sink_points)

                items.append(
                    Item(
                        class_name=cn,
                        name=r.get("mDisplayName", cn),
                        sink_points=sink_points,
                    )
                )

            elif native_class == RECIPE_NATIVE_CLASS:
                produced_in_classes = parse_class_refs(r.get("mProducedIn", ""))

                resolved_buildings = tuple(
                    buildings_by_class[bcls]
                    for bcls in produced_in_classes
                    if bcls in buildings_by_class
                )

                recipes.append(
                    Recipe(
                        class_name=cn,
                        name=r.get("mDisplayName", cn),
                        duration_s=to_float(r.get("mManufactoringDuration", "0")),
                        ingredients=parse_item_list(r.get("mIngredients", "")),
                        products=parse_item_list(r.get("mProduct", "")),
                        produced_in=produced_in_classes,
                        produced_in_buildings=resolved_buildings,
                    )
                )

    recipes_by_product: dict[str, list[Recipe]] = {}
    for rec in recipes:
        for product_class, _amount in rec.products:
            recipes_by_product.setdefault(product_class, []).append(rec)

    return items, recipes, recipes_by_product, buildings_by_class


def recipe_net_map(recipe: Recipe) -> dict[str, float]:
    net: dict[str, float] = {}

    for cls, amt in recipe.products:
        net[cls] = net.get(cls, 0.0) + float(amt)

    for cls, amt in recipe.ingredients:
        net[cls] = net.get(cls, 0.0) - float(amt)

    return net


def build_item_lookup(items: list[Item], recipes: list[Recipe]) -> tuple[dict[str, Item], list[str]]:
    items_by_class = {it.class_name: it for it in items}

    all_item_classes = set(items_by_class.keys())
    for rec in recipes:
        for cls, _ in rec.ingredients:
            all_item_classes.add(cls)
        for cls, _ in rec.products:
            all_item_classes.add(cls)

    for cls in all_item_classes:
        if cls not in items_by_class:
            items_by_class[cls] = Item(class_name=cls, name=cls, sink_points=0)

    return items_by_class, sorted(all_item_classes)


def find_zero_input_recipes(recipes: list[Recipe]) -> list[Recipe]:
    return [r for r in recipes if not r.ingredients]


def find_zero_input_sink_recipes(recipes: list[Recipe], items: list[Item]) -> list[Recipe]:
    items_by_class = {it.class_name: it for it in items}
    suspicious: list[Recipe] = []

    for rec in recipes:
        if rec.ingredients:
            continue

        total_sink_points = 0
        for cls, amt in rec.products:
            total_sink_points += items_by_class.get(cls, Item(cls, cls, 0)).sink_points * amt

        if total_sink_points > 0:
            suspicious.append(rec)

    return suspicious


def detect_profitable_zero_input_cycle(
    items: list[Item],
    recipes: list[Recipe],
    eps: float = 1e-9,
) -> SuspiciousCycleResult:
    items_by_class, item_classes = build_item_lookup(items, recipes)

    sinkable_items = [cls for cls in item_classes if items_by_class[cls].sink_points > 0]
    n_recipes = len(recipes)
    n_sink = len(sinkable_items)
    n_vars = n_recipes + n_sink

    sink_idx = {cls: i for i, cls in enumerate(sinkable_items)}
    recipe_nets = [recipe_net_map(r) for r in recipes]

    c = [0.0] * n_vars
    for cls, k in sink_idx.items():
        c[n_recipes + k] = -float(items_by_class[cls].sink_points)

    A_ub: list[list[float]] = []
    b_ub: list[float] = []

    for cls in item_classes:
        row = [0.0] * n_vars

        for j, net in enumerate(recipe_nets):
            amt = net.get(cls, 0.0)
            if abs(amt) > eps:
                row[j] = -amt

        sk = sink_idx.get(cls)
        if sk is not None:
            row[n_recipes + sk] = 1.0

        A_ub.append(row)
        b_ub.append(0.0)

    norm_row = [0.0] * n_vars
    for j in range(n_recipes):
        norm_row[j] = 1.0
    A_ub.append(norm_row)
    b_ub.append(1.0)

    bounds = [(0.0, None)] * n_vars

    res = linprog(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        return SuspiciousCycleResult(
            has_profitable_cycle=False,
            objective_value=0.0,
            recipe_usage={},
            sink_output={},
        )

    objective_value = -float(res.fun)

    recipe_usage: dict[str, float] = {}
    sink_output: dict[str, float] = {}

    for j, recipe in enumerate(recipes):
        val = float(res.x[j])
        if val > 1e-7:
            recipe_usage[recipe.class_name] = val

    for cls, k in sink_idx.items():
        val = float(res.x[n_recipes + k])
        if val > 1e-7:
            sink_output[cls] = val

    return SuspiciousCycleResult(
        has_profitable_cycle=objective_value > 1e-7,
        objective_value=objective_value,
        recipe_usage=recipe_usage,
        sink_output=sink_output,
    )


def pretty_recipe(recipe: Recipe, item_names: dict[str, str]) -> str:
    def fmt_pairs(pairs: tuple[tuple[str, float], ...]) -> str:
        if not pairs:
            return "(none)"
        return ", ".join(f"{amt:g}x {item_names.get(cls, cls)}" for cls, amt in pairs)

    def fmt_buildings(buildings: tuple[Building, ...]) -> str:
        if not buildings:
            return "(unknown)"
        return ", ".join(
            f"{b.name} [{b.class_name}, power={b.power_consumption:g}, "
            f"exp={b.power_consumption_exponent:g}, shard_slots={b.production_shard_slot_size}, "
            f"shard_boost={b.production_shard_boost_multiplier:g}]"
            for b in buildings
        )

    return (
        f"{recipe.name} ({recipe.class_name})\n"
        f"  ingredients: {fmt_pairs(recipe.ingredients)}\n"
        f"  products:    {fmt_pairs(recipe.products)}\n"
        f"  duration_s:  {recipe.duration_s:g}\n"
        f"  produced_in: {fmt_buildings(recipe.produced_in_buildings)}"
    )


if __name__ == "__main__":
    items, recipes, recipes_by_product, buildings_by_class = load_model("en-US.json")

    print(f"Loaded {len(items)} items, {len(recipes)} recipes, {len(buildings_by_class)} buildings/components.")
    print()

    item_names = {it.class_name: it.name for it in items}

    # Example: print one recipe with resolved building info
    for rec in recipes[:5]:
        print(pretty_recipe(rec, item_names))
        print()

    # Example: inspect a specific building by class name
    manufacturer = buildings_by_class.get("Build_ManufacturerMk1_C")
    if manufacturer:
        print("=== Manufacturer building info ===")
        print(manufacturer)
        print()

    zero_input = find_zero_input_recipes(recipes)
    print(f"Zero-input recipes: {len(zero_input)}")
    for rec in zero_input[:20]:
        print(pretty_recipe(rec, item_names))
        print()

    zero_input_sink = find_zero_input_sink_recipes(recipes, items)
    print(f"Zero-input recipes producing sink value: {len(zero_input_sink)}")
    for rec in zero_input_sink[:20]:
        print(pretty_recipe(rec, item_names))
        print()

    cycle_result = detect_profitable_zero_input_cycle(items, recipes)

    print("=== Profitable zero-input cycle detection ===")
    print(f"Has profitable cycle: {cycle_result.has_profitable_cycle}")
    print(f"Objective value: {cycle_result.objective_value:g}")
    print()

    if cycle_result.has_profitable_cycle:
        print("Recipes involved:")
        for recipe_class, usage in sorted(cycle_result.recipe_usage.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {recipe_class}: {usage:.6f}")
        print()

        print("Sink outputs involved:")
        for item_class, amt in sorted(cycle_result.sink_output.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {item_names.get(item_class, item_class)} ({item_class}): {amt:.6f}")