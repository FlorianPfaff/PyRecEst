import importlib.util

import pyrecest.backend as backend
import pytest
from tests.support.backend_runner import run_backend_code


def test_std_accepts_array_like_inputs_on_active_backend():
    result = backend.std([[1, 2, 3], [4, 5, 6]], axis=0, ddof=1, keepdims=True)

    assert tuple(result.shape) == (1, 3)
    assert backend.allclose(
        result,
        backend.array([[2.1213203435596424, 2.1213203435596424, 2.1213203435596424]]),
    )


@pytest.mark.backend_portable
def test_pytorch_std_accepts_array_like_inputs():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import pyrecest.backend as backend

result = backend.std([[1, 2, 3], [4, 5, 6]], axis=0, ddof=1, keepdims=True)
expected = backend.array([[2.1213203435596424, 2.1213203435596424, 2.1213203435596424]])
assert tuple(result.shape) == (1, 3)
assert result.dtype == backend.float64
assert backend.allclose(result, expected)
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
