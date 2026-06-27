"""Module entry point for ``python -m pyrecest``."""

from __future__ import annotations

import sys

from pyrecest.cli import main

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
