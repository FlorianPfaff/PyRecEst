"""Regression coverage for the PyTorch Rotation stub method surface."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from pyrecest.exceptions import BackendNotSupportedError


def _rotation_stub_class():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "pyrecest"
        / "_backend"
        / "pytorch"
        / "spatial.py"
    )
    spec = importlib.util.spec_from_file_location("pytorch_spatial_stub", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.Rotation


@pytest.mark.backend_portable
def test_pytorch_rotation_stub_exposes_vector_methods_with_backend_errors():
    rotation_class = _rotation_stub_class()

    with pytest.raises(
        BackendNotSupportedError,
        match="Rotation.from_rotvec is unavailable for backend 'pytorch'",
    ):
        rotation_class.from_rotvec([0.0, 0.0, 0.0])
    with pytest.raises(
        BackendNotSupportedError,
        match="Rotation.from_euler is unavailable for backend 'pytorch'",
    ):
        rotation_class.from_euler("xyz", [0.0, 0.0, 0.0])
    with pytest.raises(
        BackendNotSupportedError,
        match="Rotation.as_rotvec is unavailable for backend 'pytorch'",
    ):
        rotation_class.as_rotvec(None)
    with pytest.raises(
        BackendNotSupportedError,
        match="Rotation.as_euler is unavailable for backend 'pytorch'",
    ):
        rotation_class.as_euler(None, "xyz")
