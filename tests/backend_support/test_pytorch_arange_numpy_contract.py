import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_public_pytorch_arange_accepts_numpy_scalar_arguments():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import numpy as np
import torch

import pyrecest.backend as backend

assert backend.__backend_name__ == "pytorch"

values = backend.arange(np.array(3), dtype=np.float64)
assert tuple(values.shape) == (3,)
assert values.dtype == backend.float64
assert backend.to_numpy(values).tolist() == [0.0, 1.0, 2.0]

stepped = backend.arange(np.array(1), torch.tensor(6), np.array(2))
assert backend.to_numpy(stepped).tolist() == [1, 3, 5]

try:
    backend.arange(np.array([1, 2]))
except TypeError:
    pass
else:
    raise AssertionError("vector-valued arange stop was accepted")
""",
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.backend_portable
def test_raw_pytorch_arange_is_patched_under_numpy_public_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "numpy",
        """
import numpy as np
import torch

import pyrecest  # noqa: F401
import pyrecest.backend as backend
import pyrecest._backend.pytorch as pytorch_backend

assert backend.__backend_name__ == "numpy"

values = pytorch_backend.arange(np.array(4), dtype=np.dtype("int64"))
assert tuple(values.shape) == (4,)
assert values.dtype == pytorch_backend.int64
assert pytorch_backend.to_numpy(values).tolist() == [0, 1, 2, 3]

stepped = pytorch_backend.arange(torch.tensor(2), np.array(7), torch.tensor(2))
assert pytorch_backend.to_numpy(stepped).tolist() == [2, 4, 6]

try:
    pytorch_backend.arange(torch.tensor([1, 2]))
except TypeError:
    pass
else:
    raise AssertionError("vector-valued arange stop was accepted")
""",
    )

    assert result.returncode == 0, result.stderr
