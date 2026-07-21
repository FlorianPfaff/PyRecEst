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
right_vector = torch.ones(2, device=device)
right_matrix = torch.ones((2, 2), device=device)

dot_result = target.dot(torch.tensor([1.0, 2.0]), right_vector)
assert dot_result.device.type == device.type
assert tuple(dot_result.shape) == ()
if device.type != "meta":
    assert torch.allclose(dot_result.cpu(), torch.tensor(3.0))

dot_arraylike_result = target.dot([1.0, 2.0], right_vector)
assert dot_arraylike_result.device.type == device.type
assert tuple(dot_arraylike_result.shape) == ()
if device.type != "meta":
    assert torch.allclose(dot_arraylike_result.cpu(), torch.tensor(3.0))

dot_matrix_result = target.dot(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), right_matrix)
assert dot_matrix_result.device.type == device.type
assert tuple(dot_matrix_result.shape) == (2,)
if device.type != "meta":
    assert torch.allclose(dot_matrix_result.cpu(), torch.tensor([3.0, 7.0]))

outer_result = target.outer(torch.tensor([1.0, 2.0]), right_vector)
assert outer_result.device.type == device.type
assert tuple(outer_result.shape) == (2, 2)
if device.type != "meta":
    expected = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    assert torch.allclose(outer_result.cpu(), expected)

outer_matrix_result = target.outer(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), right_vector)
assert outer_matrix_result.device.type == device.type
assert tuple(outer_matrix_result.shape) == (2, 2, 2)
if device.type != "meta":
    expected = torch.tensor(
        [[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]]
    )
    assert torch.allclose(outer_matrix_result.cpu(), expected)

outer_scalar_result = target.outer(torch.tensor(2.0), right_vector)
assert outer_scalar_result.device.type == device.type
assert tuple(outer_scalar_result.shape) == (2,)
if device.type != "meta":
    assert torch.allclose(outer_scalar_result.cpu(), torch.tensor([2.0, 2.0]))

print("ok")
"""


def test_raw_pytorch_dot_outer_prefer_existing_non_cpu_device_after_import():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("numpy", _device_contract_code("raw_pytorch"))

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_public_pytorch_dot_outer_prefer_existing_non_cpu_device():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("pytorch", _device_contract_code("backend"))

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
