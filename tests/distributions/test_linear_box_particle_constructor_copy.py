import unittest

import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array, to_numpy
from pyrecest.distributions.nonperiodic.linear_box_particle_distribution import (
    LinearBoxParticleDistribution,
)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax",
    reason="JAX arrays are immutable and cannot expose mutable input aliasing.",
)
class LinearBoxParticleConstructorCopyTest(unittest.TestCase):
    def test_constructor_owns_bound_and_weight_arrays(self):
        lower = array([[0.0], [2.0]])
        upper = array([[1.0], [4.0]])
        weights = array([0.25, 0.75])

        distribution = LinearBoxParticleDistribution(lower, upper, weights)

        lower[0, 0] = 99.0
        upper[1, 0] = 99.0
        weights[0] = 1.0
        weights[1] = 0.0

        npt.assert_allclose(to_numpy(distribution.lower), [[0.0], [2.0]])
        npt.assert_allclose(to_numpy(distribution.upper), [[1.0], [4.0]])
        npt.assert_allclose(to_numpy(distribution.w), [0.25, 0.75])


if __name__ == "__main__":
    unittest.main()
