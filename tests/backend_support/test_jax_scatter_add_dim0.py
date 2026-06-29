import pytest

from tests.support.backend_runner import run_backend_code


def test_jax_scatter_add_dim_zero_uses_remaining_axis_coordinates():
    pytest.importorskip("jax")
    code = """
import pyrecest.backend as backend

values = backend.zeros((2, 3))
indices = backend.asarray([[0, 1, 0], [1, 0, 1]], dtype=backend.int64)
updates = backend.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

result = backend.scatter_add(values, 0, indices, updates)

assert backend.to_numpy(result).tolist() == [[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]]
"""
    result = run_backend_code("jax", code)

    assert result.returncode == 0, result.stderr
