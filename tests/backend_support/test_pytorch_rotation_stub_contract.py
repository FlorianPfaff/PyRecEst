"""Regression tests for the PyTorch Rotation stub API surface."""

from __future__ import annotations

import importlib.util

import pytest
from pyrecest.exceptions import BackendNotSupportedError


@pytest.mark.backend_portable
def test_pytorch_rotation_stub_exposes_matrix_methods_with_backend_error():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    from pyrecest._backend.pytorch.spatial import Rotation

    with pytest.raises(
        BackendNotSupportedError,
        match="Rotation.from_matrix is unavailable for backend 'pytorch'",
    ):
        Rotation.from_matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    rotation = object.__new__(Rotation)
    with pytest.raises(
        BackendNotSupportedError,
        match="Rotation.as_matrix is unavailable for backend 'pytorch'",
    ):
        rotation.as_matrix()
