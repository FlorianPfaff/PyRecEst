"""Public protocol contracts for PyRecEst extension points.

The :mod:`pyrecest.protocols` package contains small, runtime-checkable
capability protocols for user-defined PyRecEst components. Protocols are public
contracts: they describe the methods, attributes, and conventions that later
modules can rely on without forcing users to inherit from a specific abstract
base class.

Package-level imports are a curated convenience layer. Each protocol submodule
owns its own ``__all__``; this module re-exports those names from known protocol
submodules that are present in the installed package. Missing follow-up modules
are ignored so capability protocol pull requests can still be developed and
merged independently.
"""

from __future__ import annotations

from importlib import import_module
from typing import Final

_PROTOCOL_SUBMODULES: Final[tuple[str, ...]] = (
    "common",
    "distributions",
    "filters",
    "models",
    "conversions",
    "manifolds",
)

__all__: list[str] = []
_EXPORT_MODULE_BY_NAME: dict[str, str] = {}


def _export_submodule(module_name: str) -> None:
    """Re-export public names from one protocol submodule if it exists."""

    qualified_name = f"{__name__}.{module_name}"
    try:
        module = import_module(qualified_name)
    except ModuleNotFoundError as exc:
        if exc.name == qualified_name:
            return
        raise

    try:
        exported_names = tuple(module.__all__)
    except AttributeError as exc:
        raise AttributeError(
            f"Protocol submodule {qualified_name!r} must define __all__ for "
            "package-level re-export."
        ) from exc

    for exported_name in exported_names:
        if not isinstance(exported_name, str):
            raise TypeError(
                f"Protocol submodule {qualified_name!r} contains a non-string "
                f"__all__ entry: {exported_name!r}."
            )
        if exported_name.startswith("_"):
            raise ValueError(
                f"Protocol submodule {qualified_name!r} exports private name "
                f"{exported_name!r}."
            )

        previous_module = _EXPORT_MODULE_BY_NAME.get(exported_name)
        if previous_module is not None:
            raise RuntimeError(
                f"Duplicate pyrecest.protocols export {exported_name!r}: "
                f"{previous_module!r} and {module_name!r}."
            )

        globals()[exported_name] = getattr(module, exported_name)
        _EXPORT_MODULE_BY_NAME[exported_name] = module_name
        __all__.append(exported_name)


for _module_name in _PROTOCOL_SUBMODULES:
    _export_submodule(_module_name)

del _module_name
