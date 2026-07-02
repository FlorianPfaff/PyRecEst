"""Regression tests for PyTorch backend NumPy view conversion."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_backend_accepts_negative_stride_numpy_views():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import numpy as np
import pyrecest.backend as backend
from pyrecest._backend.pytorch._common import from_numpy

vector_view = np.arange(5.0)[::-1]
vector_tensor = from_numpy(vector_view)
assert backend.to_numpy(vector_tensor).tolist() == [4.0, 3.0, 2.0, 1.0, 0.0]

matrix_view = np.arange(6.0).reshape(2, 3)[:, ::-1]
matrix_tensor = backend.array(matrix_view)
assert backend.to_numpy(matrix_tensor).tolist() == [[2.0, 1.0, 0.0], [5.0, 4.0, 3.0]]

matrix_diagonal = backend.diagonal(matrix_view)
assert backend.to_numpy(matrix_diagonal).tolist() == [2.0, 4.0]

matrix_trace = backend.trace(matrix_view)
assert backend.to_numpy(matrix_trace).item() == 6.0
""",
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.backend_portable
def test_pytorch_backend_accepts_non_native_byte_order_numpy_arrays():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import sys

import numpy as np
import pyrecest.backend as backend
from pyrecest._backend.pytorch._common import from_numpy

non_native_prefix = ">" if sys.byteorder == "little" else "<"
source = np.array([1.25, 2.5], dtype=f"{non_native_prefix}f8")
assert not source.dtype.isnative

converted_from_numpy = from_numpy(source)
converted_array = backend.array(source)
assert converted_from_numpy.dtype == backend.float64
assert converted_array.dtype == backend.float64
assert backend.to_numpy(converted_from_numpy).tolist() == [1.25, 2.5]
assert backend.to_numpy(converted_array).tolist() == [1.25, 2.5]

source[0] = 99.0
assert backend.to_numpy(converted_from_numpy).tolist() == [1.25, 2.5]
assert backend.to_numpy(converted_array).tolist() == [1.25, 2.5]
""",
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.backend_portable
def test_pytorch_array_copies_numpy_inputs():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import numpy as np
import pyrecest.backend as backend

source = np.arange(3.0)
converted = backend.array(source)
source[0] = 99.0

assert backend.to_numpy(converted).tolist() == [0.0, 1.0, 2.0]
""",
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.backend_portable
def test_pytorch_boxed_binary_scalar_prefers_existing_cuda_tensor_device():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    result = run_backend_code(
        "pytorch",
        """
import pyrecest.backend as backend
import torch

tensor_operand = torch.ones(2, device="cuda")
result = backend.arctan2([1.0, 2.0], tensor_operand)

assert result.device.type == "cuda"
assert tuple(result.shape) == (2,)
""",
    )

    assert result.returncode == 0, result.stderr
