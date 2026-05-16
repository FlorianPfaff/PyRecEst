import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, mean, ones, random, vstack, zeros, zeros_like
from pyrecest.distributions import GaussianDistribution, LinearDiracDistribution
from pyrecest.filters.euclidean_particle_filter import EuclideanParticleFilter


class EuclideanParticleFilterTest(unittest.TestCase):
    def setUp(self):
        random.seed(42)
        self.C_prior = array([[0.7, 0.4, 0.2], [0.4, 0.6, 0.1], [0.2, 0.1, 1]])
        self.mu = array([5.0, 6.0, 7.0])
        self.prior = GaussianDistribution(self.mu, self.C_prior)
        self.sys_noise_default = GaussianDistribution(
            zeros_like(self.mu), 0.5 * self.C_prior
        )
        self.pf = EuclideanParticleFilter(n_particles=1000, dim=3)
        self.forced_mean = array([1.0, 2.0, 3.0])
        self.pf.filter_state = self.prior

    def test_prediction_accepts_in_place_set_mean_noise(self):
        pf = EuclideanParticleFilter(n_particles=3, dim=3)
        particles = array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        noise_particles = array([[0.0, 0.0, 0.0]])
        noise = LinearDiracDistribution(noise_particles)
        pf.filter_state = LinearDiracDistribution(particles)

        pf.predict_identity(noise)

        npt.assert_allclose(pf.filter_state.d, particles)
        npt.assert_allclose(noise.d, noise_particles)

    def test_predict_update_cycle_3d(self):
        for _ in range(50):
            self.pf.predict_identity(GaussianDistribution(zeros(3), self.C_prior))
            # jscpd:ignore-start
            self.assertEqual(self.pf.get_point_estimate().shape, (3,))
            for _ in range(3):
                self.pf.update_identity(self.sys_noise_default, self.forced_mean)
            # jscpd:ignore-end

        self.assertEqual(self.pf.get_point_estimate().shape, (3,))
        npt.assert_allclose(self.pf.get_point_estimate(), self.forced_mean, atol=0.15)

    def test_predict_nonlinear_nonadditive(self):
        n_noise_samples = 10
        samples = random.uniform(size=(n_noise_samples, 3))
        weights = ones(n_noise_samples) / n_noise_samples

        def f(x, w):
            return x + w

        self.pf.predict_nonlinear_nonadditive(f, samples, weights)
        est = self.pf.get_point_estimate()
        self.assertEqual(self.pf.get_point_estimate().shape, (3,))
        npt.assert_allclose(est, self.prior.mu + mean(samples, axis=0), atol=0.1)

    def test_predict_update_cycle_3d_forced_particle_pos_no_pred(self):
        force_first_particle_pos = array([1.1, 2.0, 3.0])
        self.pf.filter_state.d = vstack(
            (force_first_particle_pos, self.pf.filter_state.d[1:])
        )
        for _ in range(10):
            # jscpd:ignore-start
            self.pf.predict_identity(
                GaussianDistribution(zeros_like(self.mu), self.C_prior)
            )
            self.assertEqual(self.pf.get_point_estimate().shape, (3,))
            for _ in range(4):
                self.pf.update_identity(self.sys_noise_default, self.forced_mean)
            # jscpd:ignore-end

        self.assertEqual(self.pf.get_point_estimate().shape, (3,))
        npt.assert_allclose(
            self.pf.get_point_estimate(), force_first_particle_pos, atol=0.2
        )

    def test_update_can_skip_automatic_resampling(self):
        pf = EuclideanParticleFilter(n_particles=4, dim=1)
        particles = array([[0.0], [1.0], [2.0], [3.0]])
        pf.filter_state = LinearDiracDistribution(particles)
        pf.set_resampling_criterion(lambda _state: False)

        pf.update_nonlinear_using_likelihood(lambda x: x[:, 0] + 1.0)

        npt.assert_allclose(pf.filter_state.d, particles)
        npt.assert_allclose(pf.filter_state.w, array([0.1, 0.2, 0.3, 0.4]))

    def test_manual_resample_resets_weights(self):
        pf = EuclideanParticleFilter(n_particles=4, dim=1)
        pf.filter_state = LinearDiracDistribution(
            array([[0.0], [1.0], [2.0], [3.0]]),
            array([0.0, 0.0, 0.0, 1.0]),
        )

        pf.resample()

        npt.assert_allclose(pf.filter_state.d, array([[3.0], [3.0], [3.0], [3.0]]))
        npt.assert_allclose(pf.filter_state.w, ones(4) / 4)


if __name__ == "__main__":
    unittest.main()
