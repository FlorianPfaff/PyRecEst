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
def test_pytorch_stack_helpers_accept_array_like_sequences():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_backend

for stack_backend in (backend, raw_backend):
    to_numpy = stack_backend.to_numpy

    assert to_numpy(stack_backend.stack(([1, 2], [3, 4]))).tolist() == [[1, 2], [3, 4]]
    assert to_numpy(stack_backend.stack(([1, 2], [3, 4]), axis=1)).tolist() == [[1, 3], [2, 4]]
    mixed = stack_backend.stack((stack_backend.array([1, 2], dtype=stack_backend.int64), [3.5, 4.5]))
    assert to_numpy(mixed).tolist() == [[1.0, 2.0], [3.5, 4.5]]

    assert to_numpy(stack_backend.hstack(([1, 2], [3, 4]))).tolist() == [1, 2, 3, 4]
    assert to_numpy(stack_backend.vstack(([1, 2], [3, 4]))).tolist() == [[1, 2], [3, 4]]
    assert to_numpy(stack_backend.column_stack(([1, 2], [3, 4]))).tolist() == [[1, 3], [2, 4]]
    assert to_numpy(stack_backend.dstack(([1, 2], [3, 4]))).tolist() == [[[1, 3], [2, 4]]]

    values = backend.array([[1, 2], [3, 4]])
    assert to_numpy(stack_backend.hstack((values, [[5, 6], [7, 8]]))).tolist() == [[1, 2, 5, 6], [3, 4, 7, 8]]
    assert to_numpy(stack_backend.column_stack((values, [[5, 6], [7, 8]]))).tolist() == [[1, 2, 5, 6], [3, 4, 7, 8]]

left, right = backend.broadcast_arrays([1, 2, 3], [[10], [20]])
assert backend.to_numpy(left).tolist() == [[1, 2, 3], [1, 2, 3]]
assert backend.to_numpy(right).tolist() == [[10, 10, 10], [20, 20, 20]]

raw_left, raw_right = raw_backend.broadcast_arrays([1, 2], 3.0)
assert raw_backend.to_numpy(raw_left).tolist() == [1, 2]
assert raw_backend.to_numpy(raw_right).tolist() == [3.0, 3.0]
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, env=_backend_test_env("pytorch")
    )


@pytest.mark.backend_portable
def test_raw_pytorch_stack_helpers_accept_array_like_sequences_with_numpy_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_backend

assert getattr(backend, "__backend_name__", None) == "numpy"

assert raw_backend.to_numpy(raw_backend.stack(([1, 2], [3, 4]))).tolist() == [[1, 2], [3, 4]]
assert raw_backend.to_numpy(raw_backend.stack(([1, 2], [3, 4]), axis=1)).tolist() == [[1, 3], [2, 4]]
mixed = raw_backend.stack((raw_backend.array([1, 2], dtype=raw_backend.int64), [3.5, 4.5]))
assert raw_backend.to_numpy(mixed).tolist() == [[1.0, 2.0], [3.5, 4.5]]

assert raw_backend.to_numpy(raw_backend.hstack(([1, 2], [3, 4]))).tolist() == [1, 2, 3, 4]
assert raw_backend.to_numpy(raw_backend.vstack(([1, 2], [3, 4]))).tolist() == [[1, 2], [3, 4]]
assert raw_backend.to_numpy(raw_backend.column_stack(([1, 2], [3, 4]))).tolist() == [[1, 3], [2, 4]]
assert raw_backend.to_numpy(raw_backend.dstack(([1, 2], [3, 4]))).tolist() == [[[1, 3], [2, 4]]]

values = raw_backend.array([[1, 2], [3, 4]])
assert raw_backend.to_numpy(raw_backend.hstack((values, [[5, 6], [7, 8]]))).tolist() == [[1, 2, 5, 6], [3, 4, 7, 8]]
assert raw_backend.to_numpy(raw_backend.column_stack((values, [[5, 6], [7, 8]]))).tolist() == [[1, 2, 5, 6], [3, 4, 7, 8]]

raw_left, raw_right = raw_backend.broadcast_arrays([1, 2], 3.0)
assert raw_backend.to_numpy(raw_left).tolist() == [1, 2]
assert raw_backend.to_numpy(raw_right).tolist() == [3.0, 3.0]
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, env=_backend_test_env("numpy")
    )
