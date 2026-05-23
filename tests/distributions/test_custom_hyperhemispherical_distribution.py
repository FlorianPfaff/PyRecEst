import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, pi
from pyrecest.distributions import (
    CustomHyperhemisphericalDistribution,
    HypersphericalUniformDistribution,
)


class CustomHyperhemisphericalDistributionTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Numerical integration is only supported for the NumPy backend",
    )
    def test_from_full_hypersphere_distribution_normalizes_callable(self):
        source = HypersphericalUniformDistribution(1)

        dist = CustomHyperhemisphericalDistribution.from_distribution(source)

        self.assertEqual(dist.dim, source.dim)
        npt.assert_allclose(dist.pdf(array([1.0, 0.0])), 1.0 / pi)
        npt.assert_allclose(dist.integrate(), 1.0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
