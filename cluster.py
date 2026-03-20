#!/usr/bin/env python3

import json
import math
import argparse
from pathlib import Path


def is_close_to_int(x: float, tol: float = 1e-6) -> bool:
    return abs(x - round(x)) <= tol


def load_mode_usage(json_path: str) -> dict[str, float]:
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "mode_usage" not in data:
        raise KeyError("JSON does not contain a 'mode_usage' field.")

    mode_usage = data["mode_usage"]

    if not isinstance(mode_usage, dict):
        raise TypeError("'mode_usage' must be a JSON object.")

    cleaned = {}
    for key, value in mode_usage.items():
        try:
            cleaned[key] = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"mode_usage[{key!r}] is not numeric: {value!r}")

    return cleaned


def evaluate_divisor(mode_usage: dict[str, float], n: int, tol: float = 1e-6) -> dict:
    """
    Split every value x into:
      x = n * floor(x / n) + remainder
    or more precisely:
      x = n * base_per_module + remainder
    where base_per_module is chosen as floor(x / n).

    This produces:
      - repeated integer-heavy base module
      - remainder module
      - exact reconstruction of original totals
    """
    integer_count = 0
    near_integer_count = 0
    total_fractional_error = 0.0

    per_recipe = {}
    base_module = {}
    remainder_module = {}

    for recipe, value in mode_usage.items():
        divided = value / n
        nearest = round(divided)
        frac_error = abs(divided - nearest)

        if is_close_to_int(divided, tol):
            integer_count += 1

        if frac_error < 0.05:
            near_integer_count += 1

        total_fractional_error += frac_error

        # integer-heavy repeated base module
        base = math.floor(divided)
        remainder = value - (base * n)

        base_module[recipe] = base
        remainder_module[recipe] = remainder

        per_recipe[recipe] = {
            "original": value,
            "divided": divided,
            "nearest_integer": nearest,
            "is_integer": is_close_to_int(divided, tol),
            "base_per_module": base,
            "remainder_total": remainder,
            "reconstructed": base * n + remainder,
            "reconstruction_error": abs((base * n + remainder) - value),
        }

    max_reconstruction_error = max(
        item["reconstruction_error"] for item in per_recipe.values()
    ) if per_recipe else 0.0

    return {
        "n": n,
        "integer_count": integer_count,
        "near_integer_count": near_integer_count,
        "total_fractional_error": total_fractional_error,
        "max_reconstruction_error": max_reconstruction_error,
        "base_module": base_module,
        "remainder_module": remainder_module,
        "per_recipe": per_recipe,
    }


def sort_results(results: list[dict]) -> list[dict]:
    return sorted(
        results,
        key=lambda r: (
            -r["integer_count"],
            -r["near_integer_count"],
            r["total_fractional_error"],
            r["n"],
        ),
    )


def parse_candidates(candidate_text: str) -> list[int]:
    values = []
    for part in candidate_text.split(","):
        part = part.strip()
        if not part:
            continue
        n = int(part)
        if n <= 0:
            raise ValueError(f"All candidate divisors must be > 0, got {n}")
        values.append(n)
    if not values:
        raise ValueError("No valid candidate divisors were provided.")
    return values


def build_range(min_n: int, max_n: int) -> list[int]:
    if min_n <= 0 or max_n <= 0:
        raise ValueError("Range bounds must be > 0.")
    if min_n > max_n:
        raise ValueError("min_n must be <= max_n.")
    return list(range(min_n, max_n + 1))


def print_summary(results: list[dict], top_k: int = 10) -> None:
    print("\nBest candidates:\n")
    for result in results[:top_k]:
        print(
            f"N={result['n']:>3} | "
            f"exact integers={result['integer_count']:>3} | "
            f"near integers={result['near_integer_count']:>3} | "
            f"total fractional error={result['total_fractional_error']:.6f} | "
            f"max reconstruction error={result['max_reconstruction_error']:.12f}"
        )


def save_result(output_path: str, result: dict) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved best result to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Find a divisor N that makes as many mode_usage/N values integer as possible."
    )
    parser.add_argument(
        "json_file",
        help="Path to the input JSON file containing a 'mode_usage' object.",
    )
    parser.add_argument(
        "--candidates",
        type=str,
        default=None,
        help="Comma-separated list of candidate N values, e.g. '10,20,43'.",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=10,
        help="Minimum N to test if --candidates is not given.",
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=100,
        help="Maximum N to test if --candidates is not given.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Tolerance for deciding whether value/N counts as an integer.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many best candidates to print.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="best.json",
        help="Optional output JSON file for the best result.",
    )

    args = parser.parse_args()

    mode_usage = load_mode_usage(args.json_file)

    if args.candidates:
        candidates = parse_candidates(args.candidates)
    else:
        candidates = build_range(args.min_n, args.max_n)

    results = [evaluate_divisor(mode_usage, n, tol=args.tol) for n in candidates]
    results = sort_results(results)

    print_summary(results, top_k=args.top_k)

    best = results[0]

    print("\nBest overall candidate:\n")
    print(json.dumps(
        {
            "n": best["n"],
            "integer_count": best["integer_count"],
            "near_integer_count": best["near_integer_count"],
            "total_fractional_error": best["total_fractional_error"],
            "max_reconstruction_error": best["max_reconstruction_error"],
        },
        indent=2
    ))

    print("\nExample of base module entries (first 10):\n")
    for i, (recipe, value) in enumerate(best["base_module"].items()):
        if i >= 10:
            break
        print(f"{recipe}: {value}")

    print("\nExample of remainder module entries (first 10):\n")
    for i, (recipe, value) in enumerate(best["remainder_module"].items()):
        if i >= 10:
            break
        print(f"{recipe}: {value}")

    if args.output:
        save_result(args.output, best)


if __name__ == "__main__":
    main()