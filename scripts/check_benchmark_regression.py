#!/usr/bin/env python3
"""Check deterministic benchmark output against conservative baselines."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

JsonObject = Mapping[str, Any]


def _load_json(path: Path) -> JsonObject:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _benchmarks_by_name(payload: JsonObject, *, source: Path) -> dict[str, JsonObject]:
    benchmarks = payload.get("benchmarks")
    if not isinstance(benchmarks, list):
        raise TypeError(f"{source} must contain a 'benchmarks' list")

    result: dict[str, JsonObject] = {}
    for entry in benchmarks:
        if not isinstance(entry, Mapping):
            raise TypeError(f"{source} contains a non-object benchmark entry")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise TypeError(f"{source} contains a benchmark without a string name")
        if name in result:
            raise ValueError(f"{source} contains duplicate benchmark {name!r}")
        result[name] = entry
    return result


def _flatten_numbers(value: Any, *, path: str) -> list[float]:
    if isinstance(value, bool):
        raise TypeError(f"{path} must be numeric, not boolean")
    if isinstance(value, int | float):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{path} must be finite")
        return [number]
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        flattened: list[float] = []
        for index, item in enumerate(value):
            flattened.extend(_flatten_numbers(item, path=f"{path}[{index}]"))
        return flattened
    raise TypeError(f"{path} must be a number or nested sequence of numbers")


def _finite_nonnegative_number(
    value: Any, *, path: str
) -> tuple[float | None, str | None]:
    if isinstance(value, bool):
        return None, f"{path} must be numeric, not boolean"
    if not isinstance(value, int | float):
        return None, f"{path} must be numeric, got {value!r}"

    number = float(value)
    if not math.isfinite(number):
        return None, f"{path} must be finite"
    if number < 0.0:
        return None, f"{path} must be nonnegative"
    return number, None


def _nonnegative_integer(value: Any, *, path: str) -> tuple[int | None, str | None]:
    if isinstance(value, bool):
        return None, f"{path} must be an integer, not boolean"
    if not isinstance(value, int):
        return None, f"{path} must be an integer, got {value!r}"
    if value < 0:
        return None, f"{path} must be nonnegative"
    return value, None


def _check_numeric_sequence(
    actual: Any,
    expected: Any,
    *,
    benchmark_name: str,
    field_name: str,
    abs_tol: float,
    rel_tol: float,
) -> list[str]:
    try:
        actual_values = _flatten_numbers(actual, path=f"{benchmark_name}.{field_name}")
        expected_values = _flatten_numbers(
            expected, path=f"baseline.{benchmark_name}.{field_name}"
        )
    except (TypeError, ValueError) as exc:
        return [str(exc)]
    if len(actual_values) != len(expected_values):
        return [
            f"{benchmark_name}: {field_name} length changed from {len(expected_values)} to {len(actual_values)}"
        ]

    failures = []
    for index, (actual_value, expected_value) in enumerate(
        zip(actual_values, expected_values)
    ):
        if not math.isclose(
            actual_value, expected_value, abs_tol=abs_tol, rel_tol=rel_tol
        ):
            failures.append(
                f"{benchmark_name}: {field_name}[{index}] expected {expected_value!r}, got {actual_value!r}"
            )
    return failures


def _check_elapsed(
    actual_entry: JsonObject,
    baseline_entry: JsonObject,
    *,
    benchmark_name: str,
    max_slowdown: float,
) -> list[str]:
    if "elapsed_seconds" not in actual_entry:
        return [f"{benchmark_name}: result is missing elapsed_seconds"]

    failures = []
    elapsed, error = _finite_nonnegative_number(
        actual_entry["elapsed_seconds"], path=f"{benchmark_name}.elapsed_seconds"
    )
    if error is not None:
        return [error]

    if "max_elapsed_seconds" in baseline_entry:
        max_elapsed, error = _finite_nonnegative_number(
            baseline_entry["max_elapsed_seconds"],
            path=f"baseline.{benchmark_name}.max_elapsed_seconds",
        )
        if error is not None:
            failures.append(error)
        else:
            assert elapsed is not None
            assert max_elapsed is not None
            if elapsed > max_elapsed:
                failures.append(
                    f"{benchmark_name}: elapsed_seconds {elapsed:.6g} exceeded absolute limit {max_elapsed:.6g}"
                )

    if "elapsed_seconds" in baseline_entry:
        baseline_elapsed, error = _finite_nonnegative_number(
            baseline_entry["elapsed_seconds"],
            path=f"baseline.{benchmark_name}.elapsed_seconds",
        )
        if error is not None:
            failures.append(error)
        else:
            assert elapsed is not None
            assert baseline_elapsed is not None
            limit = baseline_elapsed * max_slowdown
            if elapsed > limit:
                failures.append(
                    f"{benchmark_name}: elapsed_seconds {elapsed:.6g} exceeded baseline {baseline_elapsed:.6g} * {max_slowdown:.6g}"
                )
    return failures


def check_benchmarks(
    actual_payload: JsonObject,
    baseline_payload: JsonObject,
    *,
    actual_path: Path,
    baseline_path: Path,
    max_slowdown: float,
    abs_tol: float,
    rel_tol: float,
) -> list[str]:
    actual = _benchmarks_by_name(actual_payload, source=actual_path)
    baseline = _benchmarks_by_name(baseline_payload, source=baseline_path)

    failures: list[str] = []
    for benchmark_name, baseline_entry in baseline.items():
        actual_entry = actual.get(benchmark_name)
        if actual_entry is None:
            failures.append(f"missing benchmark {benchmark_name!r}")
            continue

        if "iterations" in baseline_entry:
            expected_iterations, error = _nonnegative_integer(
                baseline_entry["iterations"],
                path=f"baseline.{benchmark_name}.iterations",
            )
            if error is not None:
                failures.append(error)
            elif "iterations" not in actual_entry:
                failures.append(f"{benchmark_name}: result is missing iterations")
            else:
                actual_iterations, error = _nonnegative_integer(
                    actual_entry["iterations"], path=f"{benchmark_name}.iterations"
                )
                if error is not None:
                    failures.append(error)
                elif actual_iterations != expected_iterations:
                    failures.append(
                        f"{benchmark_name}: iterations changed from {expected_iterations} to {actual_iterations}"
                    )

        failures.extend(
            _check_elapsed(
                actual_entry,
                baseline_entry,
                benchmark_name=benchmark_name,
                max_slowdown=max_slowdown,
            )
        )

        for field_name in ("final_estimate",):
            if field_name in baseline_entry:
                if field_name not in actual_entry:
                    failures.append(f"{benchmark_name}: result is missing {field_name}")
                    continue
                failures.extend(
                    _check_numeric_sequence(
                        actual_entry[field_name],
                        baseline_entry[field_name],
                        benchmark_name=benchmark_name,
                        field_name=field_name,
                        abs_tol=abs_tol,
                        rel_tol=rel_tol,
                    )
                )
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actual", type=Path, help="Benchmark JSON produced by CI")
    parser.add_argument("baseline", type=Path, help="Baseline JSON with limits")
    parser.add_argument("--max-slowdown", type=float, default=1.5)
    parser.add_argument("--abs-tol", type=float, default=1e-8)
    parser.add_argument("--rel-tol", type=float, default=1e-8)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        failures = check_benchmarks(
            _load_json(args.actual),
            _load_json(args.baseline),
            actual_path=args.actual,
            baseline_path=args.baseline,
            max_slowdown=args.max_slowdown,
            abs_tol=args.abs_tol,
            rel_tol=args.rel_tol,
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        print(f"::error::{exc}")
        return 1

    if failures:
        for failure in failures:
            print(f"::error::{failure}")
        return 1
    print("Benchmark output matches configured regression limits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
