#!/usr/bin/env python
"""Render the backend API capability matrix as Markdown.

Run from a source checkout with ``PYTHONPATH=src`` or inside the Poetry
environment.  The script intentionally reads from the same source of truth as
the CLI and tests: ``pyrecest._backend.capabilities``.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CAPABILITIES_PATH = (
    REPOSITORY_ROOT / "src" / "pyrecest" / "_backend" / "capabilities.py"
)


def _load_capabilities_module() -> ModuleType:
    """Load capability metadata without importing the full backend facade."""
    spec = importlib.util.spec_from_file_location(
        "_pyrecest_capabilities_for_docs",
        CAPABILITIES_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load capability metadata from {CAPABILITIES_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_api_backend_capabilities() -> dict[str, dict[str, str]]:
    module = _load_capabilities_module()
    capabilities: Any = getattr(module, "API_BACKEND_CAPABILITIES")
    return dict(capabilities)


def _format_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    widths = [
        max(len(row[index]) for row in [headers, *rows])
        for index in range(len(headers))
    ]
    lines = [
        "| "
        + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(headers))
        + " |",
        "|" + "|".join("-" * (width + 2) for width in widths) + "|",
    ]
    lines.extend(
        "| "
        + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))
        + " |"
        for row in rows
    )
    return lines


def render_markdown(capabilities: dict[str, dict[str, str]] | None = None) -> str:
    if capabilities is None:
        capabilities = _load_api_backend_capabilities()
    rows = []
    for api_name, row in sorted(capabilities.items()):
        rows.append(
            [
                f"`{api_name}`",
                row.get("numpy", "unknown"),
                row.get("pytorch", "unknown"),
                row.get("jax", "unknown"),
                row.get("notes", ""),
            ]
        )
    lines = _format_table(["API", "NumPy", "PyTorch", "JAX", "Notes"], rows)
    return "\n".join(lines) + "\n"


def check_document(path: Path) -> int:
    expected = render_markdown()
    actual = path.read_text(encoding="utf-8")
    if expected in actual:
        return 0

    print(
        f"{path} does not contain the generated backend API matrix. Run scripts/render_backend_api_matrix.py and update the table.",
        file=sys.stderr,
    )
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path. If omitted, write to stdout.",
    )
    parser.add_argument(
        "--check",
        type=Path,
        help="Validate that the given Markdown file contains the generated table.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    markdown = render_markdown()
    if args.check:
        return check_document(args.check)
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
