"""Regression tests for PyTorch backend quantile keyword compatibility."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


@pytest.mark.backend_portable
def test_pytorch_quantile_accepts_numpy_axis_and_keepdims_keywords():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import pyrecest.backend as backend

values = backend.asarray([[1.0, 3.0], [2.0, 5.0], [4.0, 7.0]])
median = backend.quantile(values, 0.5, axis=0)
median_keepdims = backend.quantile(values, 0.5, axis=0, keepdims=True)
list_quantile = backend.quantile([[1.0, 4.0], [3.0, 8.0]], [0.25, 0.75], axis=0)

assert backend.to_numpy(median).tolist() == [2.0, 5.0]
assert backend.to_numpy(median_keepdims).tolist() == [[2.0, 5.0]]
assert backend.to_numpy(list_quantile).tolist() == [[1.5, 5.0], [2.5, 7.0]]
""",
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.backend_portable
def test_pytorch_quantile_rejects_conflicting_method_aliases():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import pytest

import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_backend

for target in (backend, raw_backend):
    with pytest.raises(TypeError, match="method.*interpolation"):
        target.quantile(
            [0.0, 10.0],
            0.25,
            method="nearest",
            interpolation="linear",
        )

    alias_result = target.quantile(
        [0.0, 10.0],
        0.25,
        interpolation="nearest",
    )
    assert float(alias_result) == 0.0
""",
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.backend_portable
def test_pytorch_quantile_accepts_multidimensional_q_like_numpy():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import numpy as np

import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_backend

q = np.array([[0.25, 0.75], [0.1, 0.9]])
integer_values = [[1, 4], [3, 8]]
cube = np.arange(24.0).reshape(2, 3, 4)

for target in (backend, raw_backend):
    column_quantiles = target.quantile(integer_values, q, axis=0)
    tuple_axis_quantiles = target.quantile(
        cube,
        q,
        axis=(0, 2),
        keepdims=True,
    )

    np.testing.assert_allclose(
        backend.to_numpy(column_quantiles),
        np.quantile(integer_values, q, axis=0),
    )
    np.testing.assert_allclose(
        backend.to_numpy(tuple_axis_quantiles),
        np.quantile(cube, q, axis=(0, 2), keepdims=True),
    )
""",
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.backend_portable
def test_pytorch_quantile_preserves_empty_batch_dimensions():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code(
        "pytorch",
        """
import pyrecest.backend as backend
import pyrecest._backend.pytorch as raw_backend

values = backend.zeros((0, 3, 2))
scalar_quantile = backend.quantile(values, 0.5, axis=-2)
vector_quantile = backend.quantile(
    values,
    [0.25, 0.75],
    axis=1,
    keepdims=True,
)
matrix_quantile = backend.quantile(
    values,
    [[0.25, 0.75], [0.1, 0.9]],
    axis=1,
    keepdims=True,
)
raw_quantile = raw_backend.quantile(values, 0.5, dim=1, keepdim=True)

assert tuple(scalar_quantile.shape) == (0, 2)
assert tuple(vector_quantile.shape) == (2, 0, 1, 2)
assert tuple(matrix_quantile.shape) == (2, 2, 0, 1, 2)
assert tuple(raw_quantile.shape) == (0, 1, 2)
""",
    )

    assert result.returncode == 0, result.stderr
