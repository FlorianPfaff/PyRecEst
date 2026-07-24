import pytest
from tests.support.backend_runner import run_backend_code


def test_autograd_vmap_accepts_array_like_inputs():
    pytest.importorskip("autograd")

    code = """
import pyrecest.backend as backend


def add_one(row):
    return row + 1

result = backend.vmap(add_one)([[1, 2], [3, 4]])
assert backend.to_numpy(result).tolist() == [[2, 3], [4, 5]]
"""
    result = run_backend_code("autograd", code)
    assert result.returncode == 0, result.stderr


def test_autograd_vmap_rejects_scalar_arguments_with_value_error():
    pytest.importorskip("autograd")

    code = """
import pyrecest.backend as backend


def identity(value):
    return value

try:
    backend.vmap(identity)(1.0)
except ValueError as exc:
    assert "at least one dimension" in str(exc)
else:
    raise AssertionError("scalar argument unexpectedly accepted")
"""
    result = run_backend_code("autograd", code)
    assert result.returncode == 0, result.stderr


def test_autograd_vmap_remains_differentiable():
    pytest.importorskip("autograd")

    code = """
import pyrecest.backend as backend
from autograd import grad


def vmap_square_sum(x):
    return backend.sum(backend.vmap(lambda row: row * row)(x))

actual = grad(vmap_square_sum)(backend.array([1.0, 2.0]))
assert backend.to_numpy(actual).tolist() == [2.0, 4.0]
"""
    result = run_backend_code("autograd", code)
    assert result.returncode == 0, result.stderr
