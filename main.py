from pathlib import Path
import sys

from load_data import load_model, Item, Recipe, Building  # change module name if needed


def format_item_pairs(pairs: tuple[tuple[str, float], ...]) -> str:
    if not pairs:
        return "(none)"
    return ", ".join(f"{amount:g}x {class_name}" for class_name, amount in pairs)


def format_buildings(buildings: tuple[Building, ...]) -> str:
    if not buildings:
        return "(unknown)"
    return ", ".join(
        (
            f"{b.name} ({b.class_name}, "
            f"power={b.power_consumption:g}, "
            f"exp={b.power_consumption_exponent:g}, "
            f"shard_slots={b.production_shard_slot_size}, "
            f"shard_boost={b.production_shard_boost_multiplier:g})"
        )
        for b in buildings
    )


def print_basic_stats(
    items: list[Item],
    recipes: list[Recipe],
    recipes_by_product: dict[str, list[Recipe]],
    buildings_by_class: dict[str, Building],
) -> None:
    print("=== Model Summary ===")
    print(f"Items loaded:              {len(items)}")
    print(f"Recipes loaded:            {len(recipes)}")
    print(f"Products with recipes:     {len(recipes_by_product)}")
    print(f"Buildings loaded:          {len(buildings_by_class)}")
    print()


def print_top_sink_items(items: list[Item], limit: int = 10) -> None:
    print(f"=== Top {limit} Items by Sink Points ===")
    top_items = sorted(items, key=lambda x: x.sink_points, reverse=True)[:limit]

    for i, item in enumerate(top_items, start=1):
        print(f"{i:2}. {item.name} ({item.class_name}) -> sink={item.sink_points}")
    print()


def print_sample_recipes(recipes: list[Recipe], limit: int = 5) -> None:
    print(f"=== Sample {limit} Recipes ===")
    for i, recipe in enumerate(recipes[:limit], start=1):
        print(f"{i:2}. {recipe.name} ({recipe.class_name})")
        print(f"    Duration:    {recipe.duration_s:.2f}s")
        print(f"    Ingredients: {format_item_pairs(recipe.ingredients)}")
        print(f"    Products:    {format_item_pairs(recipe.products)}")

        if recipe.produced_in_buildings:
            print(f"    Produced in: {format_buildings(recipe.produced_in_buildings)}")
        elif recipe.produced_in:
            print(f"    Produced in: {', '.join(recipe.produced_in)}")
        else:
            print("    Produced in: (unknown)")
    print()


def print_recipes_for_product(product_class: str, recipes_by_product: dict[str, list[Recipe]]) -> None:
    print(f"=== Recipes Producing {product_class} ===")
    matches = recipes_by_product.get(product_class)

    if not matches:
        print("No recipes found for this product.\n")
        return

    for i, recipe in enumerate(matches, start=1):
        print(f"{i:2}. {recipe.name} ({recipe.class_name})")
        print(f"    Duration:    {recipe.duration_s:.2f}s")
        print(f"    Ingredients: {format_item_pairs(recipe.ingredients)}")
        print(f"    Products:    {format_item_pairs(recipe.products)}")

        if recipe.produced_in_buildings:
            print(f"    Produced in: {format_buildings(recipe.produced_in_buildings)}")
        elif recipe.produced_in:
            print(f"    Produced in: {', '.join(recipe.produced_in)}")
        else:
            print("    Produced in: (unknown)")
    print()


def print_lookup_examples(
    items: list[Item],
    recipes_by_product: dict[str, list[Recipe]],
    buildings_by_class: dict[str, Building],
) -> None:
    print("=== Example Lookups ===")

    items_by_class = {item.class_name: item for item in items}

    print("First 3 items:")
    for item in items[:3]:
        print(f"  - {item.class_name}: name={item.name!r}, sink_points={item.sink_points}")
    print()

    first_product = next(iter(recipes_by_product), None)
    if first_product:
        print(f"First product with recipes: {first_product}")
        recs = recipes_by_product[first_product]
        print(f"Number of recipes producing it: {len(recs)}")
        for rec in recs[:3]:
            print(f"  - {rec.class_name}")
        print()

    if first_product and first_product in items_by_class:
        item = items_by_class[first_product]
        print("Matching item for that product:")
        print(f"  name={item.name!r}, class_name={item.class_name}, sink_points={item.sink_points}")
        print()
    elif first_product:
        print("No matching item found for the example product in items list.\n")

    manufacturer = buildings_by_class.get("Build_ManufacturerMk1_C")
    if manufacturer:
        print("Example building lookup:")
        print(
            f"  {manufacturer.name} ({manufacturer.class_name}) -> "
            f"power={manufacturer.power_consumption:g}, "
            f"exp={manufacturer.power_consumption_exponent:g}, "
            f"shard_slots={manufacturer.production_shard_slot_size}, "
            f"shard_boost={manufacturer.production_shard_boost_multiplier:g}"
        )
        print()


def print_building(building_class: str, buildings_by_class: dict[str, Building]) -> None:
    print(f"=== Building Lookup: {building_class} ===")
    building = buildings_by_class.get(building_class)

    if not building:
        print("No building found for this class name.\n")
        return

    print(f"Name:                          {building.name}")
    print(f"ClassName:                     {building.class_name}")
    print(f"mPowerConsumption:             {building.power_consumption:g}")
    print(f"mPowerConsumptionExponent:     {building.power_consumption_exponent:g}")
    print(f"mProductionShardSlotSize:      {building.production_shard_slot_size}")
    print(f"mProductionShardBoostMultiplier:{building.production_shard_boost_multiplier:g}")
    print()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py <path-to-json> [product_class] [building_class]")
        print()
        print("Example:")
        print("  python main.py Docs.json")
        print("  python main.py Docs.json Desc_IronPlate_C")
        print("  python main.py Docs.json Desc_IronPlate_C Build_ManufacturerMk1_C")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    product_class = sys.argv[2] if len(sys.argv) >= 3 else None
    building_class = sys.argv[3] if len(sys.argv) >= 4 else None

    items, recipes, recipes_by_product, buildings_by_class = load_model(str(path))

    print_basic_stats(items, recipes, recipes_by_product, buildings_by_class)
    print_lookup_examples(items, recipes_by_product, buildings_by_class)
    print_top_sink_items(items, limit=10)
    print_sample_recipes(recipes, limit=5)

    if product_class:
        print_recipes_for_product(product_class, recipes_by_product)
    else:
        first_product = next(iter(recipes_by_product), None)
        if first_product:
            print_recipes_for_product(first_product, recipes_by_product)

    if building_class:
        print_building(building_class, buildings_by_class)


if __name__ == "__main__":
    main()