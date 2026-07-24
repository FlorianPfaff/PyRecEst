import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


def _zero_class_contract_code(target_module):
    return f"""
import torch
import pyrecest  # noqa: F401  # triggers backend-support compatibility patches
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch


target = {target_module}

for labels in (
    torch.empty((0,), dtype=torch.int32),
    torch.empty((2, 0), dtype=torch.int64),
):
    result = target.one_hot(labels, 0)
    assert result.dtype == torch.uint8
    assert tuple(result.shape) == tuple(labels.shape) + (0,)
    assert result.device == labels.device

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("meta")
device_labels = torch.empty((0,), dtype=torch.int32, device=device)
device_result = target.one_hot(device_labels, 0)
assert tuple(device_result.shape) == (0, 0)
assert device_result.dtype == torch.uint8
assert device_result.device.type == device.type

print("ok")
"""


def test_raw_pytorch_one_hot_handles_empty_zero_class_inputs_after_import():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("numpy", _zero_class_contract_code("raw_pytorch"))

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_public_pytorch_one_hot_handles_empty_zero_class_inputs():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("pytorch", _zero_class_contract_code("backend"))

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
