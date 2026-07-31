"""Regression tests for JAX linalg static-argument normalization."""

import pytest
from tests.support.backend_runner import run_backend_code


def test_jax_matrix_power_accepts_scalar_array_exponents():
    pytest.importorskip("jax")

    code = """
import numpy as np
import jax.numpy as jnp
import pyrecest.backend as backend
from pyrecest.backend import linalg

expected = [[1.0, 2.0], [0.0, 1.0]]
for exponent in (np.array(2), jnp.array(2)):
    result = linalg.matrix_power([[1.0, 1.0], [0.0, 1.0]], exponent)
    assert backend.to_numpy(result).tolist() == expected
"""
    result = run_backend_code("jax", code)
    assert result.returncode == 0, result.stderr


def test_jax_matrix_power_rejects_boolean_exponents():
    pytest.importorskip("jax")

    code = """
import numpy as np
import pytest
import jax.numpy as jnp
from pyrecest.backend import linalg

for exponent in (True, np.bool_(True), jnp.array(True)):
    with pytest.raises(TypeError, match="n must be an integer scalar"):
        linalg.matrix_power([[1.0, 1.0], [0.0, 1.0]], exponent)
"""
    result = run_backend_code("jax", code)
    assert result.returncode == 0, result.stderr


def test_jax_matrix_power_rejects_masked_exponents():
    pytest.importorskip("jax")

    code = """
import numpy as np
import pytest
import pyrecest.backend as backend
from pyrecest.backend import linalg

matrix = [[1.0, 1.0], [0.0, 1.0]]
for exponent in (np.ma.masked, np.ma.array(2, mask=True)):
    with pytest.raises(TypeError, match="n must be an integer scalar"):
        linalg.matrix_power(matrix, exponent)

result = linalg.matrix_power(matrix, np.ma.array(2, mask=False))
assert backend.to_numpy(result).tolist() == [[1.0, 2.0], [0.0, 1.0]]
"""
    result = run_backend_code("jax", code)
    assert result.returncode == 0, result.stderr
