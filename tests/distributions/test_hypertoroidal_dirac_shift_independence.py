import pytest

import pyrecest.backend
from pyrecest.backend import array
from pyrecest.distributions import HypertoroidalDiracDistribution


@pytest.mark.skipif(
    pyrecest.backend.__backend_name__ == "jax",
    reason="JAX arrays are immutable and cannot expose mutable aliasing.",
)
def test_shifted_weights_do_not_alias_original_distribution():
    original = HypertoroidalDiracDistribution(
        array([[0.1, 0.2], [0.3, 0.4]]),
        array([0.25, 0.75]),
    )

    shifted = original.shift(array([0.2, -0.1]))
    shifted.w[0] = 0.5

    assert float(original.w[0]) == pytest.approx(0.25)
    assert float(shifted.w[0]) == pytest.approx(0.5)
