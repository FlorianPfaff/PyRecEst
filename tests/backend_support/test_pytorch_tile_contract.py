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
def test_pytorch_tile_scalar_and_array_repetitions_match_numpy_contract():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import pyrecest.backend as backend
from pyrecest._backend import pytorch as pytorch_backend

values = backend.array([[1, 2], [3, 4]])

for tile_func, to_numpy in (
    (backend.tile, backend.to_numpy),
    (pytorch_backend.tile, pytorch_backend.to_numpy),
):
    scalar_result = tile_func(values, 2)
    assert tuple(scalar_result.shape) == (2, 4)
    assert to_numpy(scalar_result).tolist() == [[1, 2, 1, 2], [3, 4, 3, 4]]

    array_result = tile_func(values, backend.array([2, 1]))
    assert tuple(array_result.shape) == (4, 2)
    assert to_numpy(array_result).tolist() == [[1, 2], [3, 4], [1, 2], [3, 4]]

    empty_result = tile_func(values, ())
    assert tuple(empty_result.shape) == (2, 2)
    assert to_numpy(empty_result).tolist() == [[1, 2], [3, 4]]
    assert empty_result is not values

    for bad_reps in (1.5, [2.5, 1], "2", backend.array([2.5, 1.0])):
        try:
            tile_func(values, bad_reps)
        except TypeError:
            pass
        else:
            raise AssertionError(f"tile accepted non-integer repetitions {bad_reps!r}")
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, env=_backend_test_env("pytorch")
    )


@pytest.mark.backend_portable
def test_raw_pytorch_tile_matches_numpy_contract_with_numpy_public_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import pyrecest.backend as backend
from pyrecest._backend import pytorch as pytorch_backend

assert getattr(backend, "__backend_name__", None) == "numpy"
values = pytorch_backend.array([[1, 2], [3, 4]])

scalar_result = pytorch_backend.tile(values, 2)
assert tuple(scalar_result.shape) == (2, 4)
assert pytorch_backend.to_numpy(scalar_result).tolist() == [[1, 2, 1, 2], [3, 4, 3, 4]]

array_result = pytorch_backend.tile(values, pytorch_backend.array([2, 1]))
assert tuple(array_result.shape) == (4, 2)
assert pytorch_backend.to_numpy(array_result).tolist() == [[1, 2], [3, 4], [1, 2], [3, 4]]

empty_result = pytorch_backend.tile(values, ())
assert tuple(empty_result.shape) == (2, 2)
assert pytorch_backend.to_numpy(empty_result).tolist() == [[1, 2], [3, 4]]
assert empty_result is not values

for bad_reps in (1.5, [2.5, 1], "2", pytorch_backend.array([2.5, 1.0])):
    try:
        pytorch_backend.tile(values, bad_reps)
    except TypeError:
        pass
    else:
        raise AssertionError(f"raw tile accepted non-integer repetitions {bad_reps!r}")
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, env=_backend_test_env("numpy")
    )
