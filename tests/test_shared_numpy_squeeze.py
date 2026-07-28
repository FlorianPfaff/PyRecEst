import pyrecest.backend as backend
import pytest
from pyrecest.backend import array
from tests.support.backend_runner import run_backend_code


def _to_python(value):
    value = backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_shared_numpy_squeeze_accepts_tuple_axis():
    if backend.__backend_name__ not in ("numpy", "autograd"):
        pytest.skip("shared NumPy/autograd squeeze regression test")

    result = backend.squeeze(array([[[1], [2]]]), axis=(0, 2))

    assert result.shape == (2,)
    assert _to_python(result) == [1, 2]


def test_shared_numpy_squeeze_accepts_scalar_array_axis():
    if backend.__backend_name__ not in ("numpy", "autograd"):
        pytest.skip("shared NumPy/autograd squeeze regression test")
    np = pytest.importorskip("numpy")

    result = backend.squeeze(array([[1], [2]]), axis=np.array(1))

    assert result.shape == (2,)
    assert _to_python(result) == [1, 2]


def test_raw_numpy_squeeze_accepts_scalar_array_axis():
    code = """
import numpy as np
import pyrecest  # noqa: F401
import pyrecest._backend.numpy as raw_numpy

result = raw_numpy.squeeze([[1], [2]], axis=np.array(1))
assert result.shape == (2,)
assert result.tolist() == [1, 2]
print("ok")
"""
    result = run_backend_code("numpy", code)

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.parametrize("axis", [0, (0,), -2])
def test_shared_numpy_squeeze_rejects_non_singleton_axis(axis):
    if backend.__backend_name__ not in ("numpy", "autograd"):
        pytest.skip("shared NumPy/autograd squeeze regression test")

    with pytest.raises(ValueError, match="size not equal to one"):
        backend.squeeze(array([[1, 2], [3, 4]]), axis=axis)


def test_raw_numpy_squeeze_rejects_non_singleton_axis():
    code = """
import pyrecest  # noqa: F401
import pyrecest._backend.numpy as raw_numpy

for axis in (0, (0,), -2):
    try:
        raw_numpy.squeeze([[1, 2], [3, 4]], axis=axis)
    except ValueError as exc:
        assert "size not equal to one" in str(exc), str(exc)
    else:
        raise AssertionError("expected a non-singleton axis failure")
print("ok")
"""
    result = run_backend_code("numpy", code)

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.parametrize("axis", [3, -4])
def test_shared_numpy_squeeze_rejects_out_of_bounds_axis(axis):
    if backend.__backend_name__ not in ("numpy", "autograd"):
        pytest.skip("shared NumPy/autograd squeeze regression test")

    with pytest.raises(Exception) as exc_info:
        backend.squeeze(array([[[1], [2]]]), axis=axis)

    assert "out of bounds" in str(exc_info.value)


def test_raw_numpy_squeeze_rejects_out_of_bounds_axis():
    code = """
import pyrecest  # noqa: F401
import pyrecest._backend.numpy as raw_numpy

for axis in (3, -4):
    try:
        raw_numpy.squeeze([[[1], [2]]], axis=axis)
    except Exception as exc:
        assert "out of bounds" in str(exc), str(exc)
    else:
        raise AssertionError("expected an out-of-bounds axis failure")
print("ok")
"""
    result = run_backend_code("numpy", code)

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
