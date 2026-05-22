"""Command line utilities for PyRecEst."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


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
    errors: list[str] = []
    for key, expected_value in expected.items():
        if key not in actual:
            errors.append(f"{section_name}.{key} missing from scenario result")
            continue
        actual_value = actual[key]
        if isinstance(expected_value, int | float):
            delta = abs(float(actual_value) - float(expected_value))
            if delta > tolerance:
                errors.append(
                    f"{section_name}.{key} mismatch: abs_error={delta:.6g} > tolerance={tolerance:.6g}"
                )
        elif actual_value != expected_value:
            errors.append(
                f"{section_name}.{key} mismatch: expected {expected_value!r}, got {actual_value!r}"
            )
    return errors


def _cmd_run_scenario(args: argparse.Namespace) -> int:
    from pyrecest.scenarios import run_scenario

    result = run_scenario(args.config)
    print(result.to_json(indent=2))

    if args.expected is not None:
        expected = json.loads(Path(args.expected).read_text(encoding="utf-8"))
        tolerance = (
            args.tolerance
            if args.tolerance is not None
            else float(expected.get("tolerance", 1e-8))
        )
        failures: list[str] = []
        expected_estimate = expected.get("final_estimate")
        if expected_estimate is not None:
            if len(result.final_estimate) != len(expected_estimate):
                failures.append(
                    "final_estimate length mismatch: "
                    f"expected {len(expected_estimate)}, got {len(result.final_estimate)}"
                )
            else:
                max_error = _max_abs_error(result.final_estimate, expected_estimate)
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
