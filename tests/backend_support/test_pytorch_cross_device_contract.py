"""Regression tests for PyTorch ``cross`` device placement."""

from __future__ import annotations

import pyrecest.backend as backend
import pyrecest.stability  # noqa: F401 ensure compatibility patches are installed
import pytest

pytorch_backend = pytest.importorskip("pyrecest._backend.pytorch")
torch = pytest.importorskip("torch")


def _assert_cross_prefers_symbolic_operand_device(cross):
    cpu_operand = torch.tensor([1.0, 0.0, 0.0])
    symbolic_operand = torch.empty((3,), device="meta")

    result = cross(cpu_operand, symbolic_operand)

    assert result.device.type == "meta"
    assert result.shape == symbolic_operand.shape
    assert result.dtype == torch.result_type(cpu_operand, symbolic_operand)


def test_raw_pytorch_cross_prefers_meta_operand_device():
    _assert_cross_prefers_symbolic_operand_device(pytorch_backend.cross)


def test_public_pytorch_cross_prefers_meta_operand_device_when_active():
    if getattr(backend, "__backend_name__", None) != "pytorch":
        pytest.skip("public PyTorch backend is not active")

    _assert_cross_prefers_symbolic_operand_device(backend.cross)
