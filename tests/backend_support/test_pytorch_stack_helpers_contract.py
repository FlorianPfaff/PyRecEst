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


def _skip_if_no_torch_meta_device():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch  # pylint: disable=import-outside-toplevel

    try:
        torch.empty((1,), device="meta")
    except (RuntimeError, NotImplementedError, TypeError) as exc:
        pytest.skip(f"torch meta device is not available: {exc}")


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


@pytest.mark.backend_portable
@pytest.mark.parametrize("backend_name", ("pytorch", "numpy"))
def test_pytorch_stack_helpers_reject_empty_sequences_like_numpy(backend_name):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    code = """
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_backend

helpers = (raw_backend,)
if getattr(backend, "__backend_name__", None) == "pytorch":
    helpers = (backend, raw_backend)

for stack_backend in helpers:
    for helper_name in ("hstack", "vstack", "column_stack", "dstack"):
        helper = getattr(stack_backend, helper_name)
        for empty_input in ((), []):
            try:
                helper(empty_input)
            except ValueError as exc:
                assert "need at least one array to concatenate" in str(exc), str(exc)
            else:
                raise AssertionError(f"{helper_name} accepted an empty sequence")
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, env=_backend_test_env(backend_name)
    )


@pytest.mark.backend_portable
@pytest.mark.parametrize("backend_name", ("pytorch", "numpy"))
def test_pytorch_stack_helpers_preserve_non_cpu_tensor_device(backend_name):
    _skip_if_no_torch_meta_device()

    code = """
import torch

import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_backend

helpers = (raw_backend,)
if getattr(backend, "__backend_name__", None) == "pytorch":
    helpers = (backend, raw_backend)

for stack_backend in helpers:
    one_d_meta = torch.empty((2,), device="meta")

    hstack_result = stack_backend.hstack(([1, 2], one_d_meta))
    assert hstack_result.device.type == "meta"
    assert tuple(hstack_result.shape) == (4,)

    vstack_result = stack_backend.vstack(([1, 2], one_d_meta))
    assert vstack_result.device.type == "meta"
    assert tuple(vstack_result.shape) == (2, 2)

    column_stack_result = stack_backend.column_stack(([1, 2], one_d_meta))
    assert column_stack_result.device.type == "meta"
    assert tuple(column_stack_result.shape) == (2, 2)

    dstack_result = stack_backend.dstack(([1, 2], one_d_meta))
    assert dstack_result.device.type == "meta"
    assert tuple(dstack_result.shape) == (1, 2, 2)
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, env=_backend_test_env(backend_name)
    )
