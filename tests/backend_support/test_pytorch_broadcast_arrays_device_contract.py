import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


def _broadcast_arrays_device_contract_code(target_module):
    return f"""
import torch
import pyrecest  # noqa: F401  # triggers backend-support compatibility patches
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch


target = {target_module}
left, right = target.broadcast_arrays([1, 2], torch.empty((2,), device="meta"))
assert left.device.type == "meta"
assert right.device.type == "meta"
assert tuple(left.shape) == (2,)
assert tuple(right.shape) == (2,)
print("ok")
"""


def test_raw_pytorch_broadcast_arrays_prefers_existing_non_cpu_device_after_import():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "numpy",
        _broadcast_arrays_device_contract_code("raw_pytorch"),
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_public_pytorch_broadcast_arrays_prefers_existing_non_cpu_device():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        _broadcast_arrays_device_contract_code("backend"),
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
