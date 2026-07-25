"""Regression tests for JAX squeeze axis validation."""

import pytest
from tests.support.backend_runner import run_backend_code


def test_jax_squeeze_rejects_noninteger_axis_values():
    pytest.importorskip("jax")

    result = run_backend_code(
        "jax",
        """
import pyrecest.backend as backend

values = backend.array([[[1.0], [2.0]]])
for axis in (1.5, [2.0], True):
    try:
        backend.squeeze(values, axis=axis)
    except TypeError:
        pass
    else:
        raise AssertionError(f"axis {axis!r} should have raised TypeError")
print("ok")
""",
    )

    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
