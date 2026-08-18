import unittest

import numpy.testing as npt
from pyrecest.backend import array
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_watson_distribution import (
    HyperhemisphericalWatsonDistribution,
)


class HyperhemisphericalWatsonDistributionTest(unittest.TestCase):
    def test_constructor_rejects_lower_hemisphere_mode(self):
        with self.assertRaisesRegex(ValueError, "upper hyperhemisphere"):
            HyperhemisphericalWatsonDistribution(array([0.0, 0.0, -1.0]), 2.0)

    def test_set_mode_returns_new_distribution(self):
        dist = HyperhemisphericalWatsonDistribution(array([0.0, 0.0, 1.0]), 2.0)
        new_mode = array([1.0, 0.0, 0.0])

        shifted = dist.set_mode(new_mode)

        self.assertIsNot(shifted, dist)
        npt.assert_allclose(dist.mu, array([0.0, 0.0, 1.0]))
        npt.assert_allclose(shifted.mu, new_mode)

    def test_set_mode_rejects_invalid_mode(self):
        dist = HyperhemisphericalWatsonDistribution(array([0.0, 0.0, 1.0]), 2.0)

        invalid_modes = (array([1.0, 0.0]), array([0.0, 0.0, -1.0]))
        for new_mode in invalid_modes:
            with self.subTest(new_mode=new_mode):
                with self.assertRaises(ValueError):
                    dist.set_mode(new_mode)

    def test_negative_kappa_mode_is_actual_upper_hemisphere_mode(self):
        dist = HyperhemisphericalWatsonDistribution(array([0.0, 0.0, 1.0]), -2.0)

        mode = dist.mode()

        self.assertGreaterEqual(float(mode[-1]), 0.0)
        npt.assert_allclose(float(mode @ dist.mu), 0.0, atol=1e-7)
        npt.assert_allclose(float(mode @ mode), 1.0, atol=1e-7)
        self.assertGreater(float(dist.pdf(mode)), float(dist.pdf(dist.mu)))

    def test_set_mode_rejects_negative_kappa_nonunique_mode(self):
        dist = HyperhemisphericalWatsonDistribution(array([0.0, 0.0, 1.0]), -2.0)

        with self.assertRaisesRegex(ValueError, "ambiguous"):
            dist.set_mode(array([1.0, 0.0, 0.0]))

    def test_shift_rejects_noncanonical_base_direction(self):
        dist = HyperhemisphericalWatsonDistribution(array([1.0, 0.0, 0.0]), 2.0)

        with self.assertRaisesRegex(ValueError, "compatibility"):
            dist.shift(array([0.0, 0.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
