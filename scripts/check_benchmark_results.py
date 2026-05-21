#!/usr/bin/env python3
"""Validate deterministic benchmark outputs against a JSON baseline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Could not parse {path}: {exc}") from exc


def _index_benchmarks(
    payload: Mapping[str, Any], source: Path
) -> dict[str, Mapping[str, Any]]:
    benchmarks = payload.get("benchmarks")
    if not isinstance(benchmarks, list):
        raise SystemExit(f"{source} must contain a top-level 'benchmarks' list.")

    indexed: dict[str, Mapping[str, Any]] = {}
    for entry in benchmarks:
        if not isinstance(entry, Mapping):
            raise SystemExit(
                f"{source} contains a benchmark entry that is not an object: {entry!r}"
            )
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise SystemExit(
                f"{source} contains a benchmark entry without a non-empty name: {entry!r}"
            )
        if name in indexed:
            raise SystemExit(f"{source} contains duplicate benchmark entry {name!r}.")
        indexed[name] = entry
    return indexed


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _compare_nested_numbers(
    expected: Any,
    actual: Any,
    *,
    path: str,
    rtol: float,
    atol: float,
) -> list[str]:
    if _is_number(expected):
        if not _is_number(actual):
            return [f"{path}: expected numeric value {expected!r}, got {actual!r}."]
        if not math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=atol):
            return [
                f"{path}: expected {float(expected)!r}, got {float(actual)!r} with tolerances rtol={rtol}, atol={atol}."
            ]
        return []

    if isinstance(expected, Sequence) and not isinstance(expected, str):
        if not isinstance(actual, Sequence) or isinstance(actual, str):
            return [f"{path}: expected a sequence, got {actual!r}."]
        if len(expected) != len(actual):
            return [f"{path}: expected length {len(expected)}, got {len(actual)}."]

        errors: list[str] = []
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual, strict=True)
        ):
            errors.extend(
                _compare_nested_numbers(
                    expected_item,
                    actual_item,
                    path=f"{path}[{index}]",
                    rtol=rtol,
                    atol=atol,
                )
            )
        return errors

    if expected != actual:
        return [f"{path}: expected {expected!r}, got {actual!r}."]
    return []


def _runtime_errors(
    baseline_entry: Mapping[str, Any],
    result_entry: Mapping[str, Any],
    *,
    max_runtime_ratio: float | None,
) -> list[str]:
    errors: list[str] = []
    elapsed = result_entry.get("elapsed_seconds")

    max_elapsed = baseline_entry.get("max_elapsed_seconds")
    if max_elapsed is not None:
        if not _is_number(elapsed):
            errors.append(
                "elapsed_seconds is missing or not numeric in the result entry."
            )
        elif float(elapsed) > float(max_elapsed):
            errors.append(
                f"elapsed_seconds={float(elapsed):.6g} exceeds max_elapsed_seconds={float(max_elapsed):.6g}."
            )

    baseline_elapsed = baseline_entry.get("elapsed_seconds")
    if max_runtime_ratio is not None and baseline_elapsed is not None:
        if not _is_number(elapsed):
            errors.append(
                "elapsed_seconds is missing or not numeric in the result entry."
            )
        elif float(elapsed) > float(baseline_elapsed) * max_runtime_ratio:
            errors.append(
                f"elapsed_seconds={float(elapsed):.6g} exceeds baseline elapsed_seconds {float(baseline_elapsed):.6g} by more than ratio {max_runtime_ratio:.6g}."
            )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results",
        type=Path,
        help="Benchmark result JSON produced by benchmarks/basic_regressions.py.",
    )
    parser.add_argument(
        "--baseline", type=Path, required=True, help="Baseline JSON to compare against."
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-8,
        help="Relative tolerance for numeric outputs.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-8,
        help="Absolute tolerance for numeric outputs.",
    )
    parser.add_argument(
        "--max-runtime-ratio",
        type=float,
        help="Fail when runtime exceeds a baseline elapsed_seconds ratio.",
    )
    parser.add_argument(
        "--warn-only-runtime",
        action="store_true",
        help="Print runtime regressions as warnings instead of failing.",
    )
    args = parser.parse_args()

    results = _index_benchmarks(_load_json(args.results), args.results)
    baseline = _index_benchmarks(_load_json(args.baseline), args.baseline)

    errors: list[str] = []
    runtime_warnings: list[str] = []
    for name, expected_entry in baseline.items():
        actual_entry = results.get(name)
        if actual_entry is None:
            errors.append(f"Missing benchmark result {name!r}.")
            continue

        if (
            "iterations" in expected_entry
            and actual_entry.get("iterations") != expected_entry["iterations"]
        ):
            errors.append(
                f"{name}: expected iterations={expected_entry['iterations']!r}, got {actual_entry.get('iterations')!r}."
            )

        if "final_estimate" in expected_entry:
            errors.extend(
                _compare_nested_numbers(
                    expected_entry["final_estimate"],
                    actual_entry.get("final_estimate"),
                    path=f"{name}.final_estimate",
                    rtol=args.rtol,
                    atol=args.atol,
                )
            )

        runtime_messages = _runtime_errors(
            expected_entry,
            actual_entry,
            max_runtime_ratio=args.max_runtime_ratio,
        )
        if args.warn_only_runtime:
            runtime_warnings.extend(
                f"{name}: {message}" for message in runtime_messages
            )
        else:
            errors.extend(f"{name}: {message}" for message in runtime_messages)

    for warning in runtime_warnings:
        print(f"::warning::{warning}")

    if errors:
        for error in errors:
            print(f"::error::{error}")
        raise SystemExit(1)

    print(
        f"Validated {len(baseline)} benchmark baseline entr{'y' if len(baseline) == 1 else 'ies'}."
    )


if __name__ == "__main__":
    main()
