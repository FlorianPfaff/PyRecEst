import importlib.util

import pytest

import pyrecest.backend as backend
from tests.support.backend_runner import run_backend_code


def _to_python(value):
    converted = backend.to_numpy(value)
    return converted.tolist() if hasattr(converted, "tolist") else converted


@pytest.mark.backend_portable
def test_pytorch_scatter_add_accepts_array_like_inputs():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific backend contract")

    result = backend.scatter_add([10, 20, 30], 0, [0, 2], [1, 2])

    assert _to_python(result) == [11, 20, 32]


@pytest.mark.backend_portable
def test_pytorch_scatter_add_rejects_boolean_dim():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific backend contract")

    with pytest.raises(TypeError):
        backend.scatter_add([1, 2], True, [0], [1])


@pytest.mark.backend_portable
def test_pytorch_scatter_add_rejects_non_integer_indices():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific backend contract")

    np = pytest.importorskip("numpy")
    for bad_index in ([0.0], [True], np.array([0.0])):
        with pytest.raises(TypeError):
            backend.scatter_add([1, 2], 0, bad_index, [1])

    empty_result = backend.scatter_add([1, 2], 0, [], [])
    assert _to_python(empty_result) == [1, 2]


@pytest.mark.backend_portable
def test_pytorch_scatter_add_rejects_uint64_indices_outside_int64():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific backend contract")

    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    overflowing = np.array([np.iinfo(np.int64).max], dtype=np.uint64) + np.uint64(1)
    bad_indices = [overflowing]
    uint64_dtype = getattr(torch, "uint64", None)
    if uint64_dtype is not None:
        bad_indices.append(torch.tensor([2**63], dtype=uint64_dtype))

    for bad_index in bad_indices:
        with pytest.raises(ValueError, match="signed 64-bit"):
            backend.scatter_add([1, 2], 0, bad_index, [1])

    result = backend.scatter_add(
        [1, 2],
        0,
        np.array([1], dtype=np.uint64),
        [3],
    )
    assert _to_python(result) == [1, 5]


@pytest.mark.backend_portable
def test_raw_pytorch_scatter_add_rejects_non_integer_indices_with_numpy_backend():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "numpy",
        """
import numpy as np
import pyrecest  # noqa: F401  # triggers raw-backend compatibility patches
import pyrecest._backend.pytorch as raw_pytorch
import torch

for bad_index in ([0.0], [True], np.array([0.0])):
    try:
        raw_pytorch.scatter_add([1, 2], 0, bad_index, [1])
    except TypeError:
        pass
    else:
        raise AssertionError("scatter_add accepted non-integer indices")

overflowing = np.array([np.iinfo(np.int64).max], dtype=np.uint64) + np.uint64(1)
bad_unsigned_indices = [overflowing]
if getattr(torch, "uint64", None) is not None:
    bad_unsigned_indices.append(torch.tensor([2**63], dtype=torch.uint64))
for bad_index in bad_unsigned_indices:
    try:
        raw_pytorch.scatter_add([1, 2], 0, bad_index, [1])
    except ValueError as exc:
        assert "signed 64-bit" in str(exc)
    else:
        raise AssertionError("scatter_add accepted an overflowing uint64 index")

empty_result = raw_pytorch.scatter_add([1, 2], 0, [], [])
assert raw_pytorch.to_numpy(empty_result).tolist() == [1, 2]
unsigned_result = raw_pytorch.scatter_add(
    [1, 2], 0, np.array([1], dtype=np.uint64), [3]
)
assert raw_pytorch.to_numpy(unsigned_result).tolist() == [1, 5]
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
