import importlib.util
import os
import subprocess
import sys

import pytest


def _backend_subprocess_env(backend_name):
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
def test_raw_pytorch_tile_matches_numpy_when_public_backend_is_numpy():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    env = _backend_subprocess_env("numpy")

    code = """
import numpy as np
import numpy.testing as npt
import pyrecest.backend as public_backend
import pyrecest._backend.pytorch as raw_pytorch

assert public_backend.__backend_name__ == "numpy"

cases = [
    ([[1, 2], [3, 4]], 2),
    ([[1, 2], [3, 4]], [2, 1]),
    ([[1, 2], [3, 4]], ()),
    ([1, 2, 3], [2, 3]),
]

for values, repetitions in cases:
    expected = np.tile(np.asarray(values), repetitions)
    result = raw_pytorch.tile(raw_pytorch.array(values), repetitions)
    npt.assert_array_equal(raw_pytorch.to_numpy(result), expected)
    assert tuple(result.shape) == expected.shape

values = raw_pytorch.array([[1, 2], [3, 4]])
empty_result = raw_pytorch.tile(values, ())
assert empty_result is not values

for bad_reps in (1.5, [2.5, 1], "2", raw_pytorch.array([2.5, 1.0])):
    try:
        raw_pytorch.tile(values, bad_reps)
    except TypeError:
        pass
    else:
        raise AssertionError(f"tile accepted non-integer repetitions {bad_reps!r}")
"""
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


@pytest.mark.backend_portable
@pytest.mark.parametrize("backend_name", ["numpy", "pytorch"])
def test_pytorch_tile_preserves_dtype_storage_and_gradients(backend_name):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import math
import numpy as np
import numpy.testing as npt
import torch
import pyrecest.backend as public_backend
import pyrecest._backend.pytorch as raw_pytorch

tile_functions = [raw_pytorch.tile]
if public_backend.__backend_name__ == "pytorch":
    tile_functions.append(public_backend.tile)

for tile in tile_functions:
    for dtype in (torch.bool, torch.int16, torch.float32, torch.float64, torch.complex128):
        values = torch.tensor([[0, 1, 0], [1, 0, 1]], dtype=dtype).T
        original = values.clone()
        assert not values.is_contiguous()
        for reps in ((), 1, (2, 3), (0, 2)):
            result = tile(x=values, reps=reps)
            assert result.dtype == values.dtype
            assert result.device == values.device
            assert result is not values
            npt.assert_array_equal(result.numpy(), np.tile(original.numpy(), reps))
            if result.numel():
                assert result.untyped_storage().data_ptr() != values.untyped_storage().data_ptr()
                result.fill_(0)
            torch.testing.assert_close(values, original)

    for reps in ((), (1,), (2, 3), (0, 2)):
        base = torch.arange(6, dtype=torch.float64, requires_grad=True)
        values = base.reshape(2, 3).T
        original = values.detach().clone()
        result = tile(x=values, reps=reps)
        result.sum().backward()
        torch.testing.assert_close(base.grad, torch.full_like(base, math.prod(reps)))
        torch.testing.assert_close(values.detach(), original)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=_backend_subprocess_env(backend_name),
        timeout=60,
    )
