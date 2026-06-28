import pytest
from tests.support.backend_runner import run_backend_code


def test_common_torch_fallbacks_accept_negative_stride_numpy_inputs():
    pytest.importorskip("torch")

    code = """
import numpy as np
import torch

import pyrecest.backend as backend

values = np.arange(3.0)[::-1]
weights = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)

outer_result = backend.outer(values, weights)
dot_result = backend.dot(values, weights)
matvec_result = backend.matvec(np.eye(3, dtype=float)[::-1], weights)

assert backend.__backend_name__ == "pytorch"
assert torch.is_tensor(outer_result)
assert backend.to_numpy(outer_result).tolist() == [
    [2.0, 4.0, 6.0],
    [1.0, 2.0, 3.0],
    [0.0, 0.0, 0.0],
]
assert float(dot_result) == 4.0
assert backend.to_numpy(matvec_result).tolist() == [3.0, 2.0, 1.0]
"""

    result = run_backend_code("pytorch", code)

    assert result.returncode == 0, result.stderr
