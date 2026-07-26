"""Regression test for diagonal axis coercion."""

from __future__ import annotations

import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code

_CHECK = """
import pyrecest.backend as backend

matrix = backend.array([[1.0, 2.0], [3.0, 4.0]])

for axis_value in (False, True):
    try:
        backend.diagonal(matrix, axis1=axis_value)
    except TypeError:
        pass
    else:
        raise AssertionError('axis1 accepted boolean axis')

    try:
        backend.diagonal(matrix, axis2=axis_value)
    except TypeError:
        pass
    else:
        raise AssertionError('axis2 accepted boolean axis')

print('ok')
"""


@pytest.mark.backend_portable
def test_pytorch_diagonal_rejects_bool_axes():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("PyTorch is not installed")

    result = run_backend_code("pytorch", _CHECK)

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
