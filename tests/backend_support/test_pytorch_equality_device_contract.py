import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


def _device_contract_code(target_module):
    return f"""
import torch
import pyrecest  # noqa: F401  # triggers backend-support compatibility patches
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch

def _non_cpu_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("meta")

target = {target_module}
device = _non_cpu_device()

cases = [
    (
        "equal",
        torch.tensor([1.0, 2.0]),
        lambda selected_device: torch.ones(2, device=selected_device),
    ),
    (
        "less_equal",
        torch.tensor([1.0, 2.0]),
        lambda selected_device: torch.full((2,), 3.0, device=selected_device),
    ),
    (
        "less_equal",
        1.0,
        lambda selected_device: torch.ones((), device=selected_device),
    ),
]

for helper_name, left, right_factory in cases:
    result = getattr(target, helper_name)(left, right_factory(device))
    assert result.device.type == device.type
    assert result.dtype == torch.bool

print("ok")
"""


def test_raw_pytorch_equality_helpers_prefer_existing_non_cpu_device_after_import():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("numpy", _device_contract_code("raw_pytorch"))

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_public_pytorch_equality_helpers_prefer_existing_non_cpu_device():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("pytorch", _device_contract_code("backend"))

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
