import importlib.util

import pytest
from tests.support.backend_runner import run_backend_code


def _skip_without_jax():
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed")


@pytest.mark.backend_portable
def test_raw_jax_take_honors_out_under_numpy_backend():
    _skip_without_jax()

    result = run_backend_code(
        "numpy",
        """
import numpy as np
import pyrecest  # noqa: F401 - triggers backend compatibility patches
import pyrecest.backend as backend
import pyrecest._backend.jax as raw_jax

assert getattr(backend, "__backend_name__", None) == "numpy"
values = raw_jax.arange(6).reshape(2, 3)
out = raw_jax.zeros((2, 2), dtype=values.dtype)

actual = raw_jax.take(values, [0, 2], axis=1, out=out)
expected = np.array([[0, 2], [3, 5]])
np.testing.assert_array_equal(raw_jax.to_numpy(actual), expected)

try:
    raw_jax.take(values, [0, 2], axis=1, out=raw_jax.zeros((2, 3), dtype=values.dtype))
except ValueError:
    pass
else:
    raise AssertionError("raw JAX take accepted an incompatible out shape")
""",
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.backend_portable
def test_public_jax_take_honors_out_when_selected():
    _skip_without_jax()

    result = run_backend_code(
        "jax",
        """
import numpy as np
import pyrecest.backend as backend

assert getattr(backend, "__backend_name__", None) == "jax"
values = backend.arange(6).reshape(2, 3)
out = backend.zeros((2, 2), dtype=values.dtype)

actual = backend.take(values, [0, 2], axis=1, out=out)
expected = np.array([[0, 2], [3, 5]])
np.testing.assert_array_equal(backend.to_numpy(actual), expected)

try:
    backend.take(values, [0, 2], axis=1, out=backend.zeros((2, 3), dtype=values.dtype))
except ValueError:
    pass
else:
    raise AssertionError("public JAX take accepted an incompatible out shape")
""",
    )

    assert result.returncode == 0, result.stderr
