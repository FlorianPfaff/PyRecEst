#!/usr/bin/env python3
"""Report default and optional dependency groups from pyproject.toml."""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

HEAVY_DEFAULT_DEPENDENCIES = {
    "matplotlib": "plotting",
    "pyshtools": "spherical harmonics",
    "shapely": "geometry",
}


def _dependency_name(raw_name: str) -> str:
    return raw_name.split("[", 1)[0].lower().replace("_", "-")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pyproject", nargs="?", type=Path, default=Path("pyproject.toml")
    )
    parser.add_argument(
        "--fail-on-heavy-defaults",
        action="store_true",
        help="Return non-zero if known heavy dependencies are still default dependencies.",
    )
    args = parser.parse_args()

    data = tomllib.loads(args.pyproject.read_text(encoding="utf-8"))
    poetry = data["tool"]["poetry"]
    dependencies = poetry.get("dependencies", {})
    extras = poetry.get("extras", {})

    default_names = {
        _dependency_name(name)
        for name, spec in dependencies.items()
        if name != "python" and not (isinstance(spec, dict) and spec.get("optional"))
    }

    print("Default dependencies:")
    for name in sorted(default_names):
        note = HEAVY_DEFAULT_DEPENDENCIES.get(
            name.replace("-", "_")
        ) or HEAVY_DEFAULT_DEPENDENCIES.get(name)
        suffix = f"  # candidate extra: {note}" if note else ""
        print(f"- {name}{suffix}")

    print("\nExtras:")
    for name, members in sorted(extras.items()):
        print(f"- {name}: {', '.join(members)}")

    heavy_defaults = sorted(
        name
        for name in default_names
        if name in HEAVY_DEFAULT_DEPENDENCIES
        or name.replace("-", "_") in HEAVY_DEFAULT_DEPENDENCIES
    )
    if heavy_defaults and args.fail_on_heavy_defaults:
        print(
            "Heavy dependencies remain in the default install: "
            + ", ".join(heavy_defaults),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
