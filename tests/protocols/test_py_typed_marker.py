"""Tests for the package-level PEP 561 typing marker."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import pyrecest


def test_py_typed_marker_exists_in_package_root():
    """The typing marker should live beside pyrecest.__init__."""

    package_root = Path(pyrecest.__file__).resolve().parent
    marker = package_root / "py.typed"

    assert marker.is_file()


def test_py_typed_marker_is_available_as_package_resource():
    """importlib.resources should be able to see the typing marker."""

    marker = resources.files("pyrecest").joinpath("py.typed")

    assert marker.is_file()
    assert marker.read_text(encoding="utf-8").strip() == ""
