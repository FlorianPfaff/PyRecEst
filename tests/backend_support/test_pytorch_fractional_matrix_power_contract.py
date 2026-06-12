import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_fractional_matrix_power_accepts_multiple_batch_dims():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import pyrecest.backend as backend

values = backend.array(
    [
        [
            [[4.0, 0.0], [0.0, 9.0]],
            [[16.0, 0.0], [0.0, 25.0]],
            [[36.0, 0.0], [0.0, 49.0]],
        ],
        [
            [[64.0, 0.0], [0.0, 81.0]],
            [[100.0, 0.0], [0.0, 121.0]],
            [[144.0, 0.0], [0.0, 169.0]],
        ],
    ],
    dtype=backend.float64,
)

result = backend.linalg.fractional_matrix_power(values, 0.5)
expected = backend.array(
    [
        [
            [[2.0, 0.0], [0.0, 3.0]],
            [[4.0, 0.0], [0.0, 5.0]],
            [[6.0, 0.0], [0.0, 7.0]],
        ],
        [
            [[8.0, 0.0], [0.0, 9.0]],
            [[10.0, 0.0], [0.0, 11.0]],
            [[12.0, 0.0], [0.0, 13.0]],
        ],
    ],
    dtype=backend.float64,
)

assert result.shape == (2, 3, 2, 2)
assert result.dtype == backend.float64
assert backend.allclose(result, expected, atol=1e-10, rtol=1e-10)
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
