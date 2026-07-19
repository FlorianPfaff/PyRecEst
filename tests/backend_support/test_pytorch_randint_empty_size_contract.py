from __future__ import annotations

import pytest
from tests.support.backend_runner import run_backend_code


def test_pytorch_randint_accepts_empty_scalar_bounds_with_invalid_limits() -> None:
    pytest.importorskip("torch")

    code = """
import pyrecest.backend as backend

for bounds in ((5, 5), (6, 5), (-1, None), (0, None)):
    if bounds[1] is None:
        result = backend.random.randint(bounds[0], size=(2, 0))
    else:
        result = backend.random.randint(*bounds, size=(2, 0))
    assert result.shape == (2, 0)
    assert str(result.dtype).endswith("int64")
"""

    completed = run_backend_code("pytorch", code)

    assert completed.returncode == 0, completed.stderr


def test_pytorch_randint_accepts_empty_array_bounds_with_invalid_limits() -> None:
    pytest.importorskip("torch")

    code = """
import pyrecest.backend as backend

for low, high in (([5, 5], [5, 5]), ([6, 6], [5, 5])):
    result = backend.random.randint(low, high, size=(0, 2))
    assert result.shape == (0, 2)
    assert str(result.dtype).endswith("int64")
"""

    completed = run_backend_code("pytorch", code)

    assert completed.returncode == 0, completed.stderr
