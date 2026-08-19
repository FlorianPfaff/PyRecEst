import unittest

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, linalg, ones
from pyrecest.distributions import BinghamDistribution, WatsonDistribution


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax",
    "Bingham conversion tests are not supported for this backend",
)
class TestWatsonNegativeKappaBingham(unittest.TestCase):
    def test_negative_kappa_conversion_preserves_density(self):
        mu = array([1.0, 0.0, 0.0, 0.0])
        watson = WatsonDistribution(mu, -2.0)

        bingham = watson.to_bingham()

        self.assertIsInstance(bingham, BinghamDistribution)
        npt.assert_allclose(bingham.Z, array([-2.0, 0.0, 0.0, 0.0]))
        npt.assert_allclose(abs(float(bingham.M[:, 0] @ mu)), 1.0, atol=1e-7)

        xs = array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0],
            ]
        )
        xs = xs / linalg.norm(xs, axis=1).reshape((-1, 1))

        npt.assert_allclose(watson.pdf(xs), bingham.pdf(xs), rtol=1e-6, atol=1e-8)

    def test_negative_kappa_sampling_works_above_s2(self):
        dist = WatsonDistribution(array([1.0, 0.0, 0.0, 0.0]), -2.0)

        samples = dist.sample(4)

        self.assertEqual(samples.shape, (4, 4))
        npt.assert_allclose(linalg.norm(samples, axis=1), ones(4), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
