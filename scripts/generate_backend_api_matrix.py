#!/usr/bin/env python3
"""Generate the backend API matrix documentation from capability metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from pyrecest._backend.capabilities import iter_api_backend_capabilities

BACKENDS = ("numpy", "pytorch", "jax")
BACKEND_LABELS = {"numpy": "NumPy", "pytorch": "PyTorch", "jax": "JAX"}


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


def render_backend_api_matrix() -> str:
    """Return the backend API matrix page as Markdown."""
    support_rows = [
        ["`supported`", "Intended to preserve backend semantics for the listed API."],
        [
            "`bridged`",
            "Works by crossing into another numerical stack, usually NumPy/SciPy; do not assume device, dtype, or gradient preservation.",
        ],
        [
            "`partial`",
            "Numerically useful, but with documented limitations such as SciPy bridges, CPU copies, or missing gradient/device guarantees.",
        ],
        [
            "`unsupported`",
            "Should raise a clear `NotImplementedError` or be documented as unavailable for the backend.",
        ],
    ]

    api_rows = []
    for api_name, support in iter_api_backend_capabilities():
        backend_cells = [support.get(backend, "unsupported") for backend in BACKENDS]
        api_rows.append(
            [
                f"`{api_name}`",
                backend_cells[0],
                backend_cells[1],
                backend_cells[2],
                support.get("notes", ""),
            ]
        )

    lines = [
        "# Backend API Matrix",
        "",
        "PyRecEst has two related backend contracts:",
        "",
        "1. the facade-level contract for functions exposed through `pyrecest.backend`;",
        "2. the public API contract for distributions, filters, trackers, and utilities.",
        "",
        "The machine-readable source for both contracts is",
        "`src/pyrecest/_backend/capabilities.py`.",
        "",
        "To inspect the current matrix from a checkout or installed environment, run:",
        "",
        "```bash",
        "pyrecest backends --format markdown",
        "python scripts/render_backend_api_matrix.py",
        "python scripts/check_backend_api_matrix.py",
        "```",
        "",
        "The documentation table is checked against `src/pyrecest/_backend/capabilities.py`",
        "in CI so the user-facing matrix cannot silently drift from the executable metadata.",
        "",
        "## Support Levels",
        "",
        *_format_table(["Level", "Meaning"], support_rows),
        "",
        "## Public API Rows",
        "",
        "<!-- backend-api-matrix:start -->",
        *_format_table(["API", "NumPy", "PyTorch", "JAX", "Notes"], api_rows),
        "<!-- backend-api-matrix:end -->",
    ]

    lines.extend(
        [
            "",
            "When adding a new public API, add a row to the matrix, update docs if the row is",
            "user-facing, add or update the generated table, and add a focused backend test",
            "if the API is expected to be portable. CI checks that this table still reflects",
            "`src/pyrecest/_backend/capabilities.py`.",
            "",
            "## Runtime Access",
            "",
            "Use the public helper when examples or downstream packages need to inspect",
            "backend support without duplicating the table:",
            "",
            "```python",
            "from pyrecest import get_backend_support",
            "",
            'assert get_backend_support("KalmanFilter", backend="jax") == "supported"',
            "```",
            "",
            "The CLI can also render the matrix:",
            "",
            "```bash",
            "pyrecest backends --format markdown",
            "```",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, help="Write generated Markdown to this path."
    )
    args = parser.parse_args(argv)

    rendered = render_backend_api_matrix()
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
