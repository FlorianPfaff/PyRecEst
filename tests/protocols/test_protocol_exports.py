"""Tests for curated package-level protocol exports."""

from __future__ import annotations

from importlib import import_module

import pyrecest.protocols as protocols

_EXPECTED_PROTOCOL_SUBMODULES = (
    "common",
    "distributions",
    "filters",
    "models",
    "conversions",
    "manifolds",
)


def _import_optional_protocol_submodule(module_name: str):
    qualified_name = f"pyrecest.protocols.{module_name}"
    try:
        return import_module(qualified_name)
    except ModuleNotFoundError as exc:
        if exc.name == qualified_name:
            return None
        raise


def test_package_level_protocol_all_is_unique():
    assert len(protocols.__all__) == len(set(protocols.__all__))


def test_package_level_protocol_all_contains_only_public_names():
    assert all(isinstance(name, str) for name in protocols.__all__)
    assert all(not name.startswith("_") for name in protocols.__all__)


def test_package_level_exports_match_available_submodule_exports():
    expected_exports: list[str] = []

    for module_name in _EXPECTED_PROTOCOL_SUBMODULES:
        module = _import_optional_protocol_submodule(module_name)
        if module is None:
            continue

        assert hasattr(module, "__all__")
        expected_exports.extend(module.__all__)

        for exported_name in module.__all__:
            assert getattr(protocols, exported_name) is getattr(module, exported_name)

    assert protocols.__all__ == expected_exports


def test_common_protocols_remain_available_from_package_level():
    from pyrecest.protocols import ArrayLike, BackendArray, SupportsDim, SupportsInputDim
    from pyrecest.protocols.common import SupportsDim as CommonSupportsDim
    from pyrecest.protocols.common import SupportsInputDim as CommonSupportsInputDim

    assert SupportsDim is CommonSupportsDim
    assert SupportsInputDim is CommonSupportsInputDim
    assert ArrayLike is not None
    assert BackendArray is not None
