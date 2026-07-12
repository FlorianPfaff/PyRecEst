import pytest

import pyrecest.backend
from pyrecest.backend import allclose, array, copy
from pyrecest.filters.bingham_filter import BinghamFilter


@pytest.mark.skipif(
    pyrecest.backend.__backend_name__ == "jax",
    reason="BinghamFilter is not supported on the JAX backend.",
)
def test_conjugate_does_not_mutate_input_array():
    value = array([1.0, 2.0, 3.0, 4.0])
    original = copy(value)

    conjugated = BinghamFilter._conjugate(value)

    assert bool(allclose(value, original))
    assert bool(allclose(conjugated, array([1.0, -2.0, -3.0, -4.0])))
