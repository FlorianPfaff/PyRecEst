"""Smoke tests for the public protocol package seed."""

from __future__ import annotations

from pyrecest.protocols import ArrayLike, BackendArray, SupportsDim, SupportsInputDim
from pyrecest.protocols.common import SupportsDim as CommonSupportsDim


class ObjectWithDimensions:
    dim = 2
    input_dim = 3


def test_common_protocols_are_exported_from_package():
    assert SupportsDim is CommonSupportsDim
    assert ArrayLike is not None
    assert BackendArray is not None


def test_dimension_protocols_are_runtime_checkable():
    obj = ObjectWithDimensions()

    assert isinstance(obj, SupportsDim)
    assert isinstance(obj, SupportsInputDim)
