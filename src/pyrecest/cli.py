"""Command line utilities for PyRecEst."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

_INVALID_TOLERANCE_MESSAGE = "tolerance must be a non-negative finite number"
_INVALID_TOLERANCE_SCALAR_TYPES = (
    bool,
    str,
    bytes,
    bytearray,
    np.bool_,
    np.str_,
    np.bytes_,
    np.datetime64,
    np.timedelta64,
)
_INVALID_NUMERIC_DTYPE_KINDS = frozenset("mM")


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _validate_tolerance(value: Any) -> float:
    """Return a validated comparison tolerance for expected-result checks."""

    try:
        value_array = np.asarray(value)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(_INVALID_TOLERANCE_MESSAGE) from exc

    if (
        value_array.shape != ()
        or value_array.dtype.kind in _INVALID_NUMERIC_DTYPE_KINDS
    ):
        raise ValueError(_INVALID_TOLERANCE_MESSAGE)

    scalar = value_array.item()
    if isinstance(scalar, _INVALID_TOLERANCE_SCALAR_TYPES):
        raise ValueError(_INVALID_TOLERANCE_MESSAGE)

    try:
        tolerance = float(scalar)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(_INVALID_TOLERANCE_MESSAGE) from exc
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(_INVALID_TOLERANCE_MESSAGE)
    return tolerance


def _is_finite_real_number(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, _INVALID_TOLERANCE_SCALAR_TYPES)
        and math.isfinite(float(value))
    )


def _coerce_finite_real_sequence(value: Any, *, field_name: str) -> list[float]:
    if isinstance(value, (str, bytes, bytearray, dict)):
        raise ValueError(f"{field_name} must be a sequence of finite numbers")
    try:
        items = list(value)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a sequence of finite numbers") from exc

    coerced: list[float] = []
    for item in items:
        if not _is_finite_real_number(item):
            raise ValueError(f"{field_name} must contain only finite numbers")
        coerced.append(float(item))
    return coerced


def _comparison_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _comparison_value(value.tolist())
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return value
    if isinstance(value, np.generic):
        return _comparison_value(value.item())
    if isinstance(value, dict):
        return {str(key): _comparison_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_comparison_value(item) for item in value]
    return value


def _values_equal(actual_value: Any, expected_value: Any) -> bool:
    actual_value = _comparison_value(actual_value)
    expected_value = _comparison_value(expected_value)
    try:
        return bool(actual_value == expected_value)
    except (TypeError, ValueError):
        return False


def _cmd_info(_args: argparse.Namespace) -> int:
    import pyrecest
    import pyrecest.backend as backend

    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pyrecest": getattr(pyrecest, "__version__", "unknown"),
        "active_backend": getattr(backend, "__backend_name__", "unknown"),
        "dependencies": {
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
            "torch": _package_version("torch"),
            "jax": _package_version("jax"),
            "jaxlib": _package_version("jaxlib"),
            "healpy": _package_version("healpy"),
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _cmd_backends(args: argparse.Namespace) -> int:
    from pyrecest._backend.capabilities import (
        API_BACKEND_CAPABILITIES,
        BACKEND_CAPABILITIES,
    )
    from pyrecest.backend_support import format_backend_support_markdown

    payload = {"facade": BACKEND_CAPABILITIES, "api": API_BACKEND_CAPABILITIES}
    if args.format == "markdown":
        print(format_backend_support_markdown())
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _max_abs_error(actual: list[float], expected: list[float]) -> float:
    errors = [abs(float(a) - float(b)) for a, b in zip(actual, expected)]
    return max(errors) if errors else 0.0


def _check_expected_mapping(
    section_name: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    tolerance: float,
) -> list[str]:
    tolerance = _validate_tolerance(tolerance)
    errors: list[str] = []
    for key, expected_value in expected.items():
        if key not in actual:
            errors.append(f"{section_name}.{key} missing from scenario result")
            continue
        actual_value = actual[key]
        if _is_finite_real_number(expected_value):
            if not _is_finite_real_number(actual_value):
                errors.append(
                    f"{section_name}.{key} mismatch: expected finite numeric "
                    f"{expected_value!r}, got {actual_value!r}"
                )
                continue
            actual_numeric = float(actual_value)
            expected_numeric = float(expected_value)
            delta = abs(actual_numeric - expected_numeric)
            if delta > tolerance:
                errors.append(
                    f"{section_name}.{key} mismatch: abs_error={delta:.6g} > tolerance={tolerance:.6g}"
                )
        elif isinstance(expected_value, bool):
            actual_bool = _comparison_value(actual_value)
            if not isinstance(actual_bool, bool) or actual_bool is not expected_value:
                errors.append(
                    f"{section_name}.{key} mismatch: expected {expected_value!r}, got {actual_value!r}"
                )
        elif not _values_equal(actual_value, expected_value):
            errors.append(
                f"{section_name}.{key} mismatch: expected {expected_value!r}, got {actual_value!r}"
            )
    return errors


def _cmd_run_scenario(args: argparse.Namespace) -> int:
    from pyrecest.scenarios import run_scenario

    result = run_scenario(args.config)
    print(result.to_json(indent=2))

    if args.expected is not None:
        try:
            expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            print(f"failed to read expected results: {exc}", file=sys.stderr)
            return 2
        if not isinstance(expected, dict):
            print("expected results must be a JSON object", file=sys.stderr)
            return 2
        raw_tolerance = (
            args.tolerance
            if args.tolerance is not None
            else expected.get("tolerance", 1e-8)
        )
        try:
            tolerance = _validate_tolerance(raw_tolerance)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        failures: list[str] = []
        expected_estimate = expected.get("final_estimate")
        if expected_estimate is not None:
            try:
                actual_estimate_values = _coerce_finite_real_sequence(
                    result.final_estimate,
                    field_name="scenario result final_estimate",
                )
                expected_estimate_values = _coerce_finite_real_sequence(
                    expected_estimate,
                    field_name="final_estimate",
                )
            except ValueError as exc:
                failures.append(str(exc))
            else:
                if len(actual_estimate_values) != len(expected_estimate_values):
                    failures.append(
                        "final_estimate length mismatch: "
                        f"expected {len(expected_estimate_values)}, got {len(actual_estimate_values)}"
                    )
                else:
                    max_error = _max_abs_error(
                        actual_estimate_values, expected_estimate_values
                    )
                    if max_error > tolerance:
                        failures.append(
                            f"final_estimate mismatch: max_abs_error={max_error:.6g} > tolerance={tolerance:.6g}"
                        )
        if isinstance(expected.get("metrics"), dict):
            failures.extend(
                _check_expected_mapping(
                    "metrics",
                    result.metrics,
                    expected["metrics"],
                    tolerance=tolerance,
                )
            )
        if isinstance(expected.get("diagnostics"), dict):
            failures.extend(
                _check_expected_mapping(
                    "diagnostics",
                    result.diagnostics,
                    expected["diagnostics"],
                    tolerance=tolerance,
                )
            )
        if failures:
            for failure in failures:
                print(failure, file=sys.stderr)
            return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyrecest",
        description="PyRecEst command line utilities",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info_parser = subparsers.add_parser(
        "info",
        help="Print version, backend, and dependency information as JSON.",
    )
    info_parser.set_defaults(func=_cmd_info)

    backends_parser = subparsers.add_parser(
        "backends",
        help="Print backend capability metadata.",
    )
    backends_parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
        help="Output format for backend support metadata.",
    )
    backends_parser.set_defaults(func=_cmd_backends)

    scenario_parser = subparsers.add_parser(
        "run-scenario",
        help="Run a TOML scenario and print a JSON result.",
    )
    scenario_parser.add_argument(
        "config", type=Path, help="Path to scenario config.toml"
    )
    scenario_parser.add_argument(
        "--expected",
        type=Path,
        help="Optional expected-results JSON file",
    )
    scenario_parser.add_argument(
        "--tolerance",
        type=float,
        help=(
            "Tolerance for expected final estimate checks; defaults to expected JSON tolerance or 1e-8"
        ),
    )
    scenario_parser.set_defaults(func=_cmd_run_scenario)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func: Any = args.func
    return int(func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
