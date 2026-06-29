import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_gamma_accepts_array_like_inputs():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import math
import torch
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_pytorch_backend

scalar = backend.gamma(5.0)
assert torch.is_tensor(scalar)
assert tuple(scalar.shape) == ()
assert backend.allclose(scalar, backend.array(24.0))

values = backend.gamma([0.5, 1.0, 5.0])
expected = backend.array([math.sqrt(math.pi), 1.0, 24.0])
assert backend.allclose(values, expected)

raw_values = raw_pytorch_backend.gamma([1.0, 2.0, 3.0])
assert backend.allclose(raw_values, backend.array([1.0, 1.0, 2.0]))
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
