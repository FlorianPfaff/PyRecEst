import unittest

import numpy as np
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

    def test_constructor_validates_particle_count_and_dimension(self):
        valid = EuclideanParticleFilter(n_particles=np.int64(3), dim=np.int64(2))

        self.assertEqual(valid.filter_state.d.shape, (3, 2))

        invalid_arguments = (
            ("bool-n-particles", True, 1, "n_particles"),
            ("fractional-n-particles", 1.5, 1, "n_particles"),
            ("zero-n-particles", 0, 1, "n_particles"),
            ("bool-dim", 1, True, "dim"),
            ("fractional-dim", 1, 1.5, "dim"),
            ("zero-dim", 1, 0, "dim"),
        )

        for name, n_particles, dim, message in invalid_arguments:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    EuclideanParticleFilter(n_particles, dim)

    def test_prediction_accepts_in_place_set_mean_noise(self):
        pf = EuclideanParticleFilter(n_particles=3, dim=3)
        particles = array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        noise_particles = array([[0.0, 0.0, 0.0]])
        noise = LinearDiracDistribution(noise_particles)
        pf.filter_state = LinearDiracDistribution(particles)

        pf.predict_identity(noise)

        npt.assert_allclose(pf.filter_state.d, particles)
        npt.assert_allclose(noise.d, noise_particles)

    def test_predict_nonlinear_rejects_dimension_mismatched_noise(self):
        mismatched_noise = GaussianDistribution(
            array([0.0, 0.0]), array([[1.0, 0.0], [0.0, 1.0]])
        )

        with self.assertRaisesRegex(ValueError, "dimension"):
            self.pf.predict_nonlinear(lambda xs: xs, mismatched_noise)

    def test_update_identity_accepts_scalar_measurement_for_one_dimensional_noise(self):
        pf = EuclideanParticleFilter(n_particles=3, dim=1)
        particles = array([[0.0], [1.0], [2.0]])
        pf.filter_state = LinearDiracDistribution(particles)
        pf.set_resampling_criterion(lambda _state: False)
        meas_noise = GaussianDistribution(array([0.0]), array([[1.0]]))

        pf.update_identity(meas_noise, 1.0)

        npt.assert_allclose(pf.filter_state.d, particles)
        self.assertEqual(pf.filter_state.w.shape, (3,))

    def test_update_identity_rejects_wrong_measurement_shape(self):
        with self.assertRaisesRegex(ValueError, "measurement"):
            self.pf.update_identity(self.sys_noise_default, array([1.0, 2.0]))

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

    def test_predict_nonlinear_nonadditive_rejects_sample_weight_size_mismatch(self):
        samples = array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        weights = array([1.0])

        def f(x, w):
            return x + w

        with self.assertRaisesRegex(ValueError, "samples and weights"):
            self.pf.predict_nonlinear_nonadditive(f, samples, weights)

    def test_predict_nonlinear_nonadditive_rejects_invalid_weights(self):
        samples = array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

        def f(x, w):
            return x + w

        invalid_cases = (
            (array([float("nan"), 1.0]), "finite"),
            (array([float("inf"), 1.0]), "finite"),
            (array([-float("inf"), 1.0]), "finite"),
            (array([-1.0, 2.0]), "nonnegative"),
            (array([0.0, 0.0]), "positive finite total mass"),
        )
        for weights, message in invalid_cases:
            with self.subTest(weights=weights):
                with self.assertRaisesRegex(ValueError, message):
                    self.pf.predict_nonlinear_nonadditive(f, samples, weights)

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
