import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


def _label_dtype_contract_code(target_module):
    return f"""
import numpy as np
import torch
import pyrecest  # noqa: F401  # triggers backend-support compatibility patches
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch


target = {target_module}

invalid_labels = (
    [0.0, 1.0],
    np.array([0.5, 1.0], dtype=np.float64),
    True,
    np.array([True, False], dtype=bool),
    np.array([0.0 + 0.0j, 1.0 + 0.0j], dtype=np.complex128),
)
for labels in invalid_labels:
    try:
        target.one_hot(labels, 2)
    except (RuntimeError, TypeError, ValueError):
        pass
    else:
        raise AssertionError(f"non-integer labels were accepted: {{labels!r}}")

integer_labels = np.array([0, 2], dtype=np.int32)
result = target.one_hot(integer_labels, 3)
assert result.dtype == torch.uint8
assert tuple(result.shape) == (2, 3)
assert torch.equal(
    result,
    torch.tensor([[1, 0, 0], [0, 0, 1]], dtype=torch.uint8),
)

print("ok")
"""


def test_raw_pytorch_one_hot_rejects_noninteger_array_like_labels():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("numpy", _label_dtype_contract_code("raw_pytorch"))

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_public_pytorch_one_hot_rejects_noninteger_array_like_labels():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("pytorch", _label_dtype_contract_code("backend"))

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
