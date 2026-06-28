import importlib.util
import os
import subprocess
import sys

import numpy as np
import pyrecest.backend as backend
import pytest


def _shape(value):
    return tuple(backend.shape(value))


def _to_python(value):
    value = backend.to_numpy(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def test_convert_to_wider_dtype_matches_numpy_result_type_for_mixed_dtypes():
    if backend.__backend_name__ not in {"autograd", "numpy"}:
        pytest.skip("shared NumPy dtype promotion regression test")

    first, second = backend.convert_to_wider_dtype(
        [backend.array([1], dtype=np.int64), backend.array([1.5], dtype=np.float32)]
    )

    expected_dtype = np.result_type(np.dtype("int64"), np.dtype("float32"))
    assert backend.to_numpy(first).dtype == expected_dtype
    assert backend.to_numpy(second).dtype == expected_dtype
    np.testing.assert_allclose(
        backend.to_numpy(first), np.array([1], dtype=expected_dtype)
    )
    np.testing.assert_allclose(
        backend.to_numpy(second), np.array([1.5], dtype=expected_dtype)
    )


def test_sum_axis_none_keepdims_matches_numpy_contract():
    values = backend.reshape(backend.arange(6), (2, 3))

    result = backend.sum(values, keepdims=True)

    assert _shape(result) == (1, 1)
    assert _to_python(result) == [[15]]


@pytest.mark.backend_portable
def test_pytorch_sum_axis_none_keepdims_matches_numpy_contract():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = "pytorch"
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )

    code = """
import pyrecest.backend as backend
values = backend.reshape(backend.arange(6), (2, 3))
result = backend.sum(values, keepdims=True)
assert tuple(backend.shape(result)) == (1, 1)
assert backend.to_numpy(result).tolist() == [[15]]
axis_result = backend.sum(values, axis=1, keepdims=True)
assert tuple(backend.shape(axis_result)) == (2, 1)
assert backend.to_numpy(axis_result).tolist() == [[3], [12]]
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
