#!/usr/bin/env python
"""Generate the Markdown backend support table."""

from __future__ import annotations

from pathlib import Path

from backend_support_matrix import markdown_table

HEADER = """# Backend Support Matrix

This page is generated from `tools/backend_support_matrix.py`. It records a
smoke-test-backed support snapshot for selected public APIs. It is intentionally
conservative and does not prove full mathematical or numerical parity.

"""


def main() -> None:
    target = Path("docs/backend-support.md")
    target.write_text(HEADER + markdown_table() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
