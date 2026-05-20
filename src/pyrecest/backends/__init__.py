"""Explicit backend access helpers.

The public ``pyrecest.backend`` facade is still selected through the
``PYRECEST_BACKEND`` environment variable for backwards compatibility.  This
module gives tests, tools, and advanced users a direct way to inspect or use a
concrete backend module without installing another import hook or changing the
process-wide active backend.
"""

from __future__ import annotations

import importlib
from types import ModuleType

SUPPORTED_BACKENDS: tuple[str, ...] = ("numpy", "pytorch", "jax")


def get_backend(name: str) -> ModuleType:
    """Return a concrete backend implementation module by name.

    Parameters
    ----------
    name:
        One of ``"numpy"``, ``"pytorch"``, or ``"jax"``.
    """
    normalized = name.lower().strip()
    if normalized not in SUPPORTED_BACKENDS:
        supported = ", ".join(SUPPORTED_BACKENDS)
        raise ValueError(f"Unknown backend '{name}'. Supported backends: {supported}.")
    return importlib.import_module(f"pyrecest._backend.{normalized}")


__all__ = ["SUPPORTED_BACKENDS", "get_backend"]
