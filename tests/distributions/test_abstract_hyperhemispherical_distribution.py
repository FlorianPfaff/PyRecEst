import unittest

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, linalg, ones, pi, sum, zeros
from pyrecest.distributions import (
    HyperhemisphericalWatsonDistribution,
    VonMisesFisherDistribution,
)
from pyrecest.distributions.hypersphere_subset.custom_hyperhemispherical_distribution import (
    CustomHyperhemisphericalDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_uniform_distribution import (
    HyperhemisphericalUniformDistribution,
)


class TestAbstractHyperhemisphericalDistribution(unittest.TestCase):
    def setUp(self):
        self.mu_ = array([0.5, 1.0, 1.0]) / linalg.norm(array([0.5, 1.0, 1.0]))
        self.kappa_ = 2.0

    def test_get_manifold_size(self):
        """Tests get_manifold_size function with different dimensions."""
        dimensions = [(1, pi), (2, 2 * pi)]
        for dim, expected in dimensions:
            with self.subTest(dim=dim):
                hud = HyperhemisphericalUniformDistribution(dim)
                self.assertAlmostEqual(hud.get_manifold_size(), expected, delta=1e-16)

    def test_dim_one_integration_boundaries_are_upper_last_coordinate(self):
        """S1 hyperhemisphere boundaries must select cos(phi) >= 0."""
        boundaries = (
            HyperhemisphericalUniformDistribution.get_full_integration_boundaries(1)
        )

        self.assertEqual(boundaries.shape, (1, 2))
        npt.assert_allclose(boundaries, array([[-pi / 2, pi / 2]]))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_dim_one_uniform_mean_direction_points_to_upper_last_coordinate(self):
        """The numerical mean of the uniform upper semicircle is [0, 1]."""
        hud = HyperhemisphericalUniformDistribution(1)

        with self.assertWarns(UserWarning):
            mean_direction = hud.mean_direction_numerical()

        npt.assert_allclose(mean_direction, array([0.0, 1.0]), atol=1e-6)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_mode_numerical(self):
        """Tests mode_numerical."""
        watson_dist = HyperhemisphericalWatsonDistribution(self.mu_, self.kappa_)
        mode_numerical = watson_dist.mode_numerical()
        npt.assert_array_almost_equal(self.mu_, mode_numerical, decimal=6)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_sampling_hemisphere(self):
        """Test if all samples are on the correct hemisphere."""
        watson_dist = HyperhemisphericalWatsonDistribution(self.mu_, self.kappa_)
        samples = watson_dist.sample(20)
        npt.assert_array_less(-samples[:, -1], zeros(samples.shape[0]))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_sample_metropolis_hastings_basics_only(self):
        """Tests the sample_metropolis_hastings sampling"""
        vmf = VonMisesFisherDistribution(array([1.0, 0.0, 0.0]), 2.0)
        chd = CustomHyperhemisphericalDistribution(
            lambda x: vmf.pdf(x) + vmf.pdf(-x), vmf.dim
        )
        n = 10
        samples = [chd.sample_metropolis_hastings(n), chd.sample(n)]
        for s in samples:
            with self.subTest(sample=s):
                self.assertEqual(s.shape, (n, chd.input_dim))
                npt.assert_allclose(sum(s**2, axis=1), ones(n), rtol=1e-10)

    def test_watson_set_mode_returns_new_distribution(self):
        mu = array([0.0, 0.0, 1.0])
        new_mode = array([0.0, 1.0, 0.0])
        dist = HyperhemisphericalWatsonDistribution(mu, self.kappa_)

        shifted = dist.set_mode(new_mode)

        self.assertIsNot(shifted, dist)
        npt.assert_allclose(dist.mu, mu)
        npt.assert_allclose(shifted.mu, new_mode)

    def test_watson_shift_accepts_flat_canonical_mu_and_returns_new_distribution(self):
        mu = array([0.0, 0.0, 1.0])
        new_mode = array([0.0, 1.0, 0.0])
        dist = HyperhemisphericalWatsonDistribution(mu, self.kappa_)

        shifted = dist.shift(new_mode)

        self.assertIsNot(shifted, dist)
        npt.assert_allclose(dist.mu, mu)
        npt.assert_allclose(shifted.mu, new_mode)


if __name__ == "__main__":
    unittest.main()
