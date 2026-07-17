import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


def _matvec_device_contract_code(target_module):
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

vector = torch.ones(2, device=device)
plain_result = target.matvec(torch.eye(2), vector)
assert plain_result.device.type == device.type
assert tuple(plain_result.shape) == (2,)
if device.type != "meta":
    assert torch.allclose(plain_result.cpu(), torch.ones(2))

array_like_result = target.matvec([[1.0, 2.0]], vector)
assert array_like_result.device.type == device.type
assert tuple(array_like_result.shape) == (1,)
if device.type != "meta":
    assert torch.allclose(array_like_result.cpu(), torch.tensor([3.0]))

batched_result = target.matvec(torch.eye(2).repeat(2, 1, 1), vector)
assert batched_result.device.type == device.type
assert tuple(batched_result.shape) == (2, 2)
if device.type != "meta":
    assert torch.allclose(batched_result.cpu(), torch.ones((2, 2)))

print("ok")
"""


def test_raw_pytorch_matvec_prefers_existing_non_cpu_device_after_import():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("numpy", _matvec_device_contract_code("raw_pytorch"))

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_public_pytorch_matvec_prefers_existing_non_cpu_device():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("pytorch", _matvec_device_contract_code("backend"))

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
