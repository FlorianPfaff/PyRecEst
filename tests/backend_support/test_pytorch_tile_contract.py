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


@pytest.mark.backend_portable
@pytest.mark.parametrize("backend_name", ["numpy", "pytorch"])
def test_pytorch_tile_shapes_repetitions_and_validation(backend_name):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import numpy as np
import numpy.testing as npt
import torch
import pyrecest.backend as public_backend
import pyrecest._backend.pytorch as raw_pytorch

tile_functions = [raw_pytorch.tile]
if public_backend.__backend_name__ == "pytorch":
    tile_functions.append(public_backend.tile)

inputs = [
    3,
    [1, 2, 3],
    [[1, 2], [3, 4]],
    torch.tensor(3, dtype=torch.int16),
    torch.tensor([1, 2, 3], dtype=torch.int32),
    torch.arange(6, dtype=torch.float32).reshape(2, 3),
    torch.arange(12, dtype=torch.float64).reshape(3, 4).T,
]
repetition_cases = [
    (2, 2),
    (np.int64(2), 2),
    (np.array(2, dtype=np.int32), 2),
    (torch.tensor(2, dtype=torch.int32), 2),
    ([2, 1], (2, 1)),
    ((2, 1), (2, 1)),
    (np.array([2, 1], dtype=np.int32), (2, 1)),
    (torch.tensor([2, 1], dtype=torch.int64), (2, 1)),
    ([], ()),
    ((), ()),
    (torch.tensor([], dtype=torch.int64), ()),
    (0, 0),
    ((2, 0), (2, 0)),
    ((2, 1, 3), (2, 1, 3)),
]

for tile in tile_functions:
    for values in inputs:
        expected_input = values.numpy().copy() if torch.is_tensor(values) else np.array(values)
        for reps, numpy_reps in repetition_cases:
            expected = np.tile(expected_input, numpy_reps)
            result = tile(x=values, reps=reps)
            assert isinstance(result, torch.Tensor)
            npt.assert_array_equal(result.numpy(), expected)
            assert tuple(result.shape) == expected.shape
            if torch.is_tensor(values):
                assert result.dtype == values.dtype
                assert result.device == values.device
                npt.assert_array_equal(values.numpy(), expected_input)
            if torch.is_tensor(reps):
                npt.assert_array_equal(reps.numpy(), np.asarray(numpy_reps))
            else:
                npt.assert_array_equal(np.asarray(reps), np.asarray(numpy_reps))

    values = torch.tensor([[1, 2], [3, 4]])
    invalid_reps = (
        None, 1.0, 1.5, float("nan"), float("inf"), 1 + 0j, "2",
        [2, 1.0], [[2, 1]], np.array([2.0, 1.0]),
        torch.tensor(2.0), torch.tensor([2.0, 1.0]), torch.tensor([[2, 1]]),
    )
    for reps in invalid_reps:
        try:
            tile(x=values, reps=reps)
        except TypeError:
            pass
        else:
            raise AssertionError(f"tile accepted invalid repetitions {reps!r}")

    for reps in (-1, [-1, 2], np.array([2, -1]), torch.tensor(-1)):
        try:
            tile(x=values, reps=reps)
        except ValueError:
            pass
        else:
            raise AssertionError(f"tile accepted negative repetitions {reps!r}")
    npt.assert_array_equal(values.numpy(), [[1, 2], [3, 4]])
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=_backend_test_env(backend_name),
        timeout=60,
    )
