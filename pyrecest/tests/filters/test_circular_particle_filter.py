import unittest

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import arange, array, linspace, pi, random
from pyrecest.distributions import (
    HypertoroidalDiracDistribution,
    WrappedNormalDistribution,
)
from pyrecest.distributions.circle.circular_dirac_distribution import (
    CircularDiracDistribution,
)
from pyrecest.distributions.circle.circular_uniform_distribution import (
    CircularUniformDistribution,
)
from pyrecest.distributions.circle.von_mises_distribution import VonMisesDistribution
from pyrecest.filters.circular_particle_filter import CircularParticleFilter


class CircularParticleFilterTest(unittest.TestCase):
    def setUp(self):
        self.n_particles = 30
        self.filter = CircularParticleFilter(self.n_particles)
        self.dist = self.filter.filter_state
        self.wn = WrappedNormalDistribution(array(1.3), array(0.8))

    def test_estimate(self):
        npt.assert_array_almost_equal(self.dist.trigonometric_moment(1), 0.0)

    def test_setting_state(self):
        # sanity check
        self.filter.filter_state = self.dist
        dist1 = self.filter.filter_state
        self.assertIsInstance(dist1, HypertoroidalDiracDistribution)
        self.assertEqual(dist1.dim, 1)
        npt.assert_array_almost_equal(self.dist.d, dist1.d)
        npt.assert_array_almost_equal(self.dist.w, dist1.w)

    def test_sampling(self):
        positions = arange(0, 1.1, 0.1)
        dist3 = CircularDiracDistribution(positions)
        random.seed(0)
        num_samples = 20
        samples = dist3.sample(num_samples)
        self.assertEqual(samples.shape, (num_samples,))
        for i in range(num_samples):
            self.assertIn(samples[i], positions)

    def test_prediction(self):
        # test prediction
        self.filter.filter_state = self.dist

        def f(x):
            return x

        self.filter.predict_nonlinear(f, self.wn)
        dist2 = self.filter.filter_state
        self.assertIsInstance(dist2, HypertoroidalDiracDistribution)
        self.assertEqual(dist2.dim, 1)

        self.filter.filter_state = self.dist
        self.filter.predict_identity(self.wn)
        dist2_identity = self.filter.filter_state
        self.assertIsInstance(dist2_identity, HypertoroidalDiracDistribution)
        self.assertEqual(dist2_identity.dim, 1)
        npt.assert_array_almost_equal(dist2.w, dist2_identity.w)

    def test_nonlinear_prediction_without_noise(self):
        # nonlinear test without noise
        self.filter.filter_state = self.dist

        def f(x):
            return x**2

        no_noise = CircularDiracDistribution(array([0.0]))
        self.filter.predict_nonlinear(f, no_noise)
        predicted = self.filter.filter_state
        self.assertIsInstance(predicted, HypertoroidalDiracDistribution)
        dist_f = self.dist.apply_function(f)
        npt.assert_array_almost_equal(predicted.d, dist_f.d, decimal=10)
        npt.assert_array_almost_equal(predicted.w, dist_f.w, decimal=10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on current backend",
    )
    def test_update(self):
        # test update
        random.seed(0)
        self.filter.filter_state = self.dist

        def h(x):
            return x

        z = array(0.0)

        def likelihood(z, x):
            return self.wn.pdf(z - h(x))

        self.filter.update_nonlinear_using_likelihood(likelihood, z)
        dist3a = self.filter.filter_state
        self.assertIsInstance(dist3a, CircularDiracDistribution)
        self.filter.filter_state = self.dist
        self.filter.update_identity(self.wn, z)
        dist3b = self.filter.filter_state
        self.assertIsInstance(dist3b, CircularDiracDistribution)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on current backend",
    )
    def test_association_likelihood(self):
        dist = CircularDiracDistribution(
            array([1.0, 2.0, 3.0]), array([1 / 3, 1 / 3, 1 / 3])
        )
        pf = CircularParticleFilter(3)
        pf.filter_state = dist

        self.assertAlmostEqual(
            pf.association_likelihood(CircularUniformDistribution()),
            1.0 / (2.0 * pi),
            places=10,
        )
        self.assertGreater(
            pf.association_likelihood(VonMisesDistribution(array(2), array(1))),
            1.0 / (2.0 * pi),
        )

        self.filter.filter_state = CircularDiracDistribution(linspace(0.0, 1.1, 30))

        def likelihood1(_, x):
            return (x == 1.1) + 0.0  # To convert it to double regardless of the backend

        self.filter.update_nonlinear_using_likelihood(likelihood1, 42)
        estimation = self.filter.filter_state
        self.assertIsInstance(estimation, CircularDiracDistribution)
        for i in range(len(estimation.d)):
            self.assertEqual(estimation.d[i], 1.1)

        # test update with single parameter likelihood
        random.seed(0)
        self.filter.filter_state = self.dist
        wn = WrappedNormalDistribution(array(1.3), array(0.8))

        def likelihood2(x):
            return wn.pdf(-x)

        self.filter.update_nonlinear_using_likelihood(likelihood2)
        dist3c = self.filter.filter_state
        self.assertIsInstance(dist3c, HypertoroidalDiracDistribution)


if __name__ == "__main__":
    unittest.main()
