import importlib.util
import os
import subprocess
import sys

import pytest


def _backend_test_env(backend_name):
    env = os.environ.copy()
    env["PYRECEST_BACKEND"] = backend_name
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path
        if not env.get("PYTHONPATH")
        else os.pathsep.join([src_path, env["PYTHONPATH"]])
    )
    return env


@pytest.mark.backend_portable
def test_pytorch_broadcast_to_shape_inputs_match_numpy_contract():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import numpy as np
import pyrecest.backend as backend
import pyrecest._backend.pytorch as pytorch_backend

values = backend.array([1, 2])

for broadcast_to, to_numpy in (
    (backend.broadcast_to, backend.to_numpy),
    (pytorch_backend.broadcast_to, pytorch_backend.to_numpy),
):
    tensor_shape_result = broadcast_to(values, backend.array([2, 2]))
    assert tuple(tensor_shape_result.shape) == (2, 2)
    assert to_numpy(tensor_shape_result).tolist() == [[1, 2], [1, 2]]

    numpy_shape_result = broadcast_to(values, np.array([2, 2]))
    assert tuple(numpy_shape_result.shape) == (2, 2)
    assert to_numpy(numpy_shape_result).tolist() == [[1, 2], [1, 2]]

    scalar_result = broadcast_to(3, backend.array(2))
    assert tuple(scalar_result.shape) == (2,)
    assert to_numpy(scalar_result).tolist() == [3, 3]

    try:
        broadcast_to(values, (-1, 2))
    except ValueError:
        pass
    else:
        raise AssertionError("broadcast_to accepted a negative broadcast dimension")

    for invalid_shape in (True, (True,), np.array(True), np.array([True]), backend.array([True])):
        try:
            broadcast_to(values, invalid_shape)
        except TypeError:
            pass
        else:
            raise AssertionError(f"broadcast_to accepted boolean shape {invalid_shape!r}")
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, env=_backend_test_env("pytorch")
    )


@pytest.mark.backend_portable
def test_raw_pytorch_broadcast_to_shape_inputs_with_numpy_public_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import numpy as np
import pyrecest.backend as backend
import pyrecest._backend.pytorch as pytorch_backend

assert getattr(backend, "__backend_name__", None) == "numpy"
values = pytorch_backend.array([1, 2])

for shape in (pytorch_backend.array([2, 2]), np.array([2, 2])):
    result = pytorch_backend.broadcast_to(values, shape)
    assert tuple(result.shape) == (2, 2)
    assert pytorch_backend.to_numpy(result).tolist() == [[1, 2], [1, 2]]

scalar_result = pytorch_backend.broadcast_to(3, pytorch_backend.array(2))
assert tuple(scalar_result.shape) == (2,)
assert pytorch_backend.to_numpy(scalar_result).tolist() == [3, 3]

try:
    pytorch_backend.broadcast_to(values, (-1, 2))
except ValueError:
    pass
else:
    raise AssertionError("raw broadcast_to accepted a negative broadcast dimension")

for invalid_shape in (True, (True,), np.array(True), np.array([True]), pytorch_backend.array([True])):
    try:
        pytorch_backend.broadcast_to(values, invalid_shape)
    except TypeError:
        pass
    else:
        raise AssertionError(f"raw broadcast_to accepted boolean shape {invalid_shape!r}")
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, env=_backend_test_env("numpy")
    )
