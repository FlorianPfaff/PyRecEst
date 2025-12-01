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


if __name__ == "__main__":
    unittest.main()
