import importlib.util

import pytest

from tests.support.backend_runner import run_backend_code

pytestmark = pytest.mark.backend_portable


def test_public_pytorch_dot_matches_matrix_product_shape():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    code = """
import pyrecest.backend as backend

left = backend.array([[1.0, 2.0], [3.0, 4.0]])
right = backend.array([[5.0, 6.0], [7.0, 8.0]])
result = backend.dot(left, right)
assert tuple(result.shape) == (2, 2)
assert backend.to_numpy(result).tolist() == [[19.0, 22.0], [43.0, 50.0]]
print("ok")
"""
    result = run_backend_code("pytorch", code)

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
