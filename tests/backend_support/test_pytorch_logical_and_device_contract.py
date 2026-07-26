"""Regression tests for PyTorch ``logical_and`` device placement."""

from __future__ import annotations

import pyrecest.backend as backend
import pyrecest.backend_support  # noqa: F401 ensure compatibility patches are installed
import pytest

pytorch_backend = pytest.importorskip("pyrecest._backend.pytorch")
torch = pytest.importorskip("torch")


def _assert_logical_and_prefers_symbolic_operand_device(logical_and):
    cpu_operand = torch.tensor([True, False])
    symbolic_operand = torch.empty((2,), dtype=torch.bool, device="meta")

    result = logical_and(cpu_operand, symbolic_operand)

    assert result.device.type == "meta"
    assert result.shape == symbolic_operand.shape
    assert result.dtype == torch.bool


def test_raw_pytorch_logical_and_prefers_meta_operand_device():
    _assert_logical_and_prefers_symbolic_operand_device(pytorch_backend.logical_and)


def test_public_pytorch_logical_and_prefers_meta_operand_device_when_active():
    if getattr(backend, "__backend_name__", None) != "pytorch":
        pytest.skip("public PyTorch backend is not active")

    _assert_logical_and_prefers_symbolic_operand_device(backend.logical_and)
