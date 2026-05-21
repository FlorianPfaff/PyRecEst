#!/usr/bin/env python
"""Check that the documented backend API matrix matches capability metadata.

The checker intentionally loads ``src/pyrecest/_backend/capabilities.py`` from a
file path instead of importing ``pyrecest``. That keeps it usable in lightweight
documentation jobs that may not install the package's numerical dependencies.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

BACKEND_COLUMNS = ("numpy", "pytorch", "jax")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_capability_module(source_path: Path | None = None) -> ModuleType:
    """Load the backend capability module without importing the package."""
    capabilities_path = (
        source_path
        or _repo_root() / "src" / "pyrecest" / "_backend" / "capabilities.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_pyrecest_backend_capabilities", capabilities_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot load backend capability metadata from {capabilities_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_documented_matrix(path: Path) -> dict[str, dict[str, str]]:
    """Parse the public API backend matrix table from ``docs/backend-api-matrix.md``."""
    rows: dict[str, dict[str, str]] = {}
    in_public_api_table = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            if in_public_api_table:
                break
            continue

        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells == ["API", "NumPy", "PyTorch", "JAX", "Notes"]:
            in_public_api_table = True
            continue
        if not in_public_api_table:
            continue
        if len(cells) == 5 and all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        if len(cells) != 5:
            continue

        api_name = cells[0].strip("`")
        rows[api_name] = {
            "numpy": cells[1],
            "pytorch": cells[2],
            "jax": cells[3],
            "notes": cells[4],
        }

    return rows


def _normalize_expected_row(row: dict[str, str]) -> dict[str, str]:
    return {
        "numpy": row.get("numpy", "unknown"),
        "pytorch": row.get("pytorch", "unknown"),
        "jax": row.get("jax", "unknown"),
        "notes": row.get("notes", ""),
    }


def validate_documented_matrix(
    documented: dict[str, dict[str, str]],
    capabilities: dict[str, dict[str, str]],
    support_levels: tuple[str, ...],
) -> list[str]:
    """Return validation errors for mismatches between docs and metadata."""
    errors: list[str] = []
    documented_names = set(documented)
    capability_names = set(capabilities)

    for missing in sorted(capability_names - documented_names):
        errors.append(f"docs/backend-api-matrix.md is missing API row `{missing}`")
    for extra in sorted(documented_names - capability_names):
        errors.append(f"docs/backend-api-matrix.md contains unknown API row `{extra}`")

    for api_name in sorted(documented_names & capability_names):
        documented_row = documented[api_name]
        expected_row = _normalize_expected_row(capabilities[api_name])
        for backend_name in BACKEND_COLUMNS:
            expected = expected_row[backend_name]
            observed = documented_row[backend_name]
            if expected not in support_levels:
                errors.append(
                    f"metadata row `{api_name}` has invalid {backend_name} support level `{expected}`"
                )
            if observed != expected:
                errors.append(
                    f"docs/backend-api-matrix.md row `{api_name}` has {backend_name}={observed!r}; expected {expected!r}"
                )
        if documented_row["notes"] != expected_row["notes"]:
            errors.append(
                f"docs/backend-api-matrix.md row `{api_name}` has notes {documented_row['notes']!r}; expected {expected_row['notes']!r}"
            )

    return errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--docs",
        type=Path,
        default=_repo_root() / "docs" / "backend-api-matrix.md",
        help="Path to docs/backend-api-matrix.md.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=_repo_root() / "src" / "pyrecest" / "_backend" / "capabilities.py",
        help="Path to src/pyrecest/_backend/capabilities.py.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    module = load_capability_module(args.source)
    documented = parse_documented_matrix(args.docs)

    if not documented:
        print(f"No public API matrix table found in {args.docs}", file=sys.stderr)
        return 1

    errors = validate_documented_matrix(
        documented,
        dict(module.API_BACKEND_CAPABILITIES),
        tuple(module.BACKEND_SUPPORT_LEVELS),
    )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"Backend API matrix is synchronized with {args.source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
