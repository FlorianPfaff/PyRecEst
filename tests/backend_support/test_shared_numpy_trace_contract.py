import pytest
from tests.support.backend_runner import run_backend_code


def test_autograd_trace_accepts_numpy_style_kwargs():
    pytest.importorskip("autograd")

    code = """
import numpy as np
import pyrecest.backend as backend

values_np = np.array(
    [
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
    ]
)
values = backend.asarray(values_np)

actual = backend.to_numpy(
    backend.trace(values, offset=1, axis1=1, axis2=2, dtype=backend.float64)
)
expected = np.trace(values_np, offset=1, axis1=1, axis2=2, dtype=np.float64)
assert actual.dtype == expected.dtype
np.testing.assert_allclose(actual, expected)

default_actual = backend.to_numpy(backend.trace(values))
default_expected = np.trace(values_np, axis1=-2, axis2=-1)
np.testing.assert_allclose(default_actual, default_expected)
"""

    result = run_backend_code("autograd", code)
    assert result.returncode == 0, result.stderr
