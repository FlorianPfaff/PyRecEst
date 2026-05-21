#!/usr/bin/env python3
"""Validate and render the public API stability registry."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
API_REGISTRY_PATH = REPOSITORY_ROOT / "src" / "pyrecest" / "api_registry.py"
CAPABILITIES_PATH = REPOSITORY_ROOT / "src" / "pyrecest" / "_backend" / "capabilities.py"


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_registry() -> tuple[dict[str, dict[str, str]], tuple[str, ...]]:
    module = _load_module(API_REGISTRY_PATH, "_pyrecest_api_registry_for_docs")
    registry: Any = getattr(module, "PUBLIC_API_REGISTRY")
    categories: Any = getattr(module, "PUBLIC_API_CATEGORIES")
    return dict(registry), tuple(categories)


def _load_backend_capabilities() -> dict[str, dict[str, str]]:
    module = _load_module(CAPABILITIES_PATH, "_pyrecest_capabilities_for_registry")
    capabilities: Any = getattr(module, "API_BACKEND_CAPABILITIES")
    return dict(capabilities)


def validate_registry() -> list[str]:
    registry, categories = _load_registry()
    backend_capabilities = _load_backend_capabilities()
    errors: list[str] = []

    if not registry:
        errors.append("PUBLIC_API_REGISTRY must not be empty")

    for api_name, row in sorted(registry.items()):
        if not api_name:
            errors.append("registry contains an empty API name")
        module = row.get("module")
        if not isinstance(module, str) or not module.startswith("pyrecest"):
            errors.append(f"{api_name}: module must be a pyrecest module path")
        category = row.get("category")
        if category not in categories:
            errors.append(f"{api_name}: unknown category {category!r}")
        notes = row.get("notes")
        if not isinstance(notes, str) or not notes.strip():
            errors.append(f"{api_name}: notes must be non-empty")
        backend_contract = row.get("backend_contract")
        if backend_contract and backend_contract not in backend_capabilities:
            errors.append(f"{api_name}: unknown backend contract {backend_contract!r}")

    for api_name in sorted(set(backend_capabilities) - set(registry)):
        errors.append(f"{api_name}: backend capability row is missing from PUBLIC_API_REGISTRY")

    return errors


def render_markdown() -> str:
    registry, _ = _load_registry()
    headers = ["API", "Module", "Category", "Backend contract", "Notes"]
    rows = []
    for api_name, row in sorted(registry.items()):
        rows.append(
            [
                f"`{api_name}`",
                f"`{row['module']}`",
                row["category"],
                f"`{row.get('backend_contract', '')}`",
                row.get("notes", ""),
            ]
        )

    widths = [max(len(values[index]) for values in [headers, *rows]) for index in range(len(headers))]

    def format_row(values: list[str]) -> str:
        cells = [f" {value.ljust(widths[index])} " for index, value in enumerate(values)]
        return "|" + "|".join(cells) + "|"

    separator = "|" + "|".join("-" * (width + 2) for width in widths) + "|"
    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines) + "\n"


def check_document(path: Path) -> int:
    expected = render_markdown()
    actual = path.read_text(encoding="utf-8")
    if expected in actual:
        return 0
    print(
        f"{path} does not contain the generated public API registry. Run scripts/check_public_api_registry.py and update the table.",
        file=sys.stderr,
    )
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    parser.add_argument("--check", type=Path, help="Validate a Markdown document.")
    return parser


def main(argv: list[str] | None = None) -> int:
    errors = validate_registry()
    if errors:
        for error in errors:
            print(f"::error::{error}")
        return 1

    args = build_parser().parse_args(argv)
    if args.check:
        return check_document(args.check)

    markdown = render_markdown()
    if args.output:
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
