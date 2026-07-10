import importlib.util

import pytest

from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


def _take_index_contract_code(target_module):
    return f"""
import numpy as np
import torch
import pyrecest  # noqa: F401  # triggers backend compatibility patches
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch

values = torch.tensor([[1, 2, 3], [4, 5, 6]])
target = {target_module}

invalid_indices = (
    np.asarray([1.5]),
    np.asarray([1], dtype=object),
    np.asarray(["1"]),
    np.asarray([np.timedelta64(1, "ns")]),
    torch.tensor([1.5]),
    torch.tensor([1.0 + 0.0j]),
)
for indices in invalid_indices:
    try:
        target.take(values, indices, axis=1)
    except TypeError:
        pass
    else:
        raise AssertionError(
            f"target.take accepted invalid index array with dtype {{indices.dtype}}"
        )

integer_result = target.take(values, np.asarray([1], dtype=np.int64), axis=1)
assert target.to_numpy(integer_result).tolist() == [[2], [5]]

boolean_result = target.take(values, np.asarray([True]), axis=1)
assert target.to_numpy(boolean_result).tolist() == [[2], [5]]

torch_integer_result = target.take(values, torch.tensor([1]), axis=1)
assert target.to_numpy(torch_integer_result).tolist() == [[2], [5]]

torch_boolean_result = target.take(values, torch.tensor([True]), axis=1)
assert target.to_numpy(torch_boolean_result).tolist() == [[2], [5]]

# NumPy accepts Python floating-point index lists by converting the sequence to
# integer indices internally. Preserve that established array-like behavior.
python_sequence_result = target.take(values, [1.5], axis=1)
assert target.to_numpy(python_sequence_result).tolist() == [[2], [5]]

print("ok")
"""


def test_raw_pytorch_take_rejects_invalid_index_arrays_with_numpy_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "numpy",
        _take_index_contract_code("raw_pytorch"),
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_public_pytorch_take_rejects_invalid_index_arrays():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        _take_index_contract_code("backend"),
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
