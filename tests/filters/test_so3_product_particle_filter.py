import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, cos, diag, exp, linalg, ones, pi, random, sin, stack
from pyrecest.filters import (
    HyperhemisphereCartProdParticleFilter,
    SO3ProductParticleFilter,
)

ATOL = 1e-6


def z_quaternion(angle):
    return array([0.0, 0.0, sin(angle / 2.0), cos(angle / 2.0)])


def x_quaternion(angle):
    return array([sin(angle / 2.0), 0.0, 0.0, cos(angle / 2.0)])


class SO3ProductParticleFilterTest(unittest.TestCase):
    def test_initializes_identity_particles(self):
        filt = SO3ProductParticleFilter(n_particles=5, num_rotations=2)

        self.assertIsInstance(filt, HyperhemisphereCartProdParticleFilter)
        self.assertEqual(filt.filter_state.dim_hemisphere, 3)
        self.assertEqual(filt.filter_state.n_hemispheres, 2)
        self.assertEqual(filt.particles.shape, (5, 2, 4))
        self.assertEqual(filt.filter_state.d.shape, (5, 8))
        npt.assert_allclose(linalg.norm(filt.particles, axis=-1), ones((5, 2)))
        npt.assert_allclose(filt.weights, ones(5) / 5)
        npt.assert_allclose(filt.effective_sample_size(), 5.0)

    def test_set_particles_normalizes_and_canonicalizes(self):
        filt = SO3ProductParticleFilter(n_particles=2, num_rotations=2)
        particles = array(
            [
                [[0.0, 0.0, 0.0, -2.0], [0.0, 0.0, 1.0, 1.0]],
                [[0.0, 0.0, 0.0, 3.0], [-1.0, 0.0, 0.0, -1.0]],
            ]
        )

        filt.set_particles(particles, weights=array([1.0, 3.0]))

        npt.assert_allclose(linalg.norm(filt.particles, axis=-1), ones((2, 2)))
        self.assertGreaterEqual(float(filt.particles[0, 0, -1]), 0.0)
        self.assertGreaterEqual(float(filt.particles[1, 1, -1]), 0.0)
        npt.assert_allclose(filt.weights, array([0.25, 0.75]))

    def test_predict_with_tangent_delta_rotates_each_component(self):
        filt = SO3ProductParticleFilter(n_particles=3, num_rotations=2)
        tangent_delta = array([0.0, 0.0, pi / 2.0, pi / 4.0, 0.0, 0.0])

        filt.predict_with_tangent_delta(tangent_delta)

        expected = array([z_quaternion(pi / 2.0), x_quaternion(pi / 4.0)])
        npt.assert_allclose(filt.particles[0], expected, atol=ATOL)
        npt.assert_allclose(filt.particles[1], expected, atol=ATOL)

    def test_predict_identity_with_full_tangent_noise_covariance(self):
        random.seed(0)
        filt = SO3ProductParticleFilter(n_particles=4, num_rotations=2)

        filt.predict_identity(diag(array([0.01, 0.01, 0.01, 0.02, 0.02, 0.02])))

        self.assertEqual(filt.particles.shape, (4, 2, 4))
        npt.assert_allclose(linalg.norm(filt.particles, axis=-1), ones((4, 2)))

    def test_update_with_geodesic_likelihood_prefers_matching_particle(self):
        filt = SO3ProductParticleFilter(n_particles=2, num_rotations=1)
        filt.set_particles(
            array([[[0.0, 0.0, 0.0, 1.0]], [z_quaternion(pi / 2.0)]]),
            weights=array([0.5, 0.5]),
        )

        ess = filt.update_with_geodesic_likelihood(
            array([[0.0, 0.0, 0.0, 1.0]]),
            noise_std=0.2,
            resample=False,
        )

        self.assertLess(float(ess), 2.0)
        self.assertGreater(float(filt.weights[0]), float(filt.weights[1]))

    def test_update_with_log_likelihood_matches_likelihood_update(self):
        likelihood_values = array([0.25, 1.0])
        log_likelihood_values = array([-1.3862943611198906, 0.0])

        likelihood_filter = SO3ProductParticleFilter(n_particles=2, num_rotations=1)
        log_likelihood_filter = SO3ProductParticleFilter(n_particles=2, num_rotations=1)

        likelihood_filter.update_with_likelihood(
            lambda _: likelihood_values, resample=False
        )
        log_likelihood_filter.update_with_log_likelihood(
            lambda _: log_likelihood_values, resample=False
        )

        npt.assert_allclose(log_likelihood_filter.weights, likelihood_filter.weights)

    def test_update_with_log_likelihood_is_underflow_safe(self):
        filt = SO3ProductParticleFilter(n_particles=2, num_rotations=1)

        filt.update_with_log_likelihood(array([-1000.0, -1001.0]), resample=False)

        npt.assert_allclose(filt.weights, array([1.0, exp(-1.0)]) / (1.0 + exp(-1.0)))

    def test_update_mask_ignores_unobserved_rotations(self):
        filt = SO3ProductParticleFilter(n_particles=2, num_rotations=2)
        filt.set_particles(
            stack(
                [
                    array([[0.0, 0.0, 0.0, 1.0], z_quaternion(0.0)]),
                    array([[0.0, 0.0, 0.0, 1.0], z_quaternion(pi / 2.0)]),
                ],
                axis=0,
            )
        )

        filt.update_with_geodesic_likelihood(
            array([[0.0, 0.0, 0.0, 1.0], z_quaternion(0.0)]),
            noise_std=0.2,
            mask=array([1.0, 0.0]),
            resample=False,
        )

        npt.assert_allclose(filt.weights, array([0.5, 0.5]), atol=ATOL)

    def test_log_likelihood_update_supports_zero_likelihoods(self):
        filt = SO3ProductParticleFilter(n_particles=3, num_rotations=1)

        ess = filt.update_with_log_likelihood(
            array([0.0, -1000.0, -float("inf")]),
            resample=False,
        )

        self.assertLess(float(ess), 2.0)
        self.assertGreater(float(filt.weights[0]), 0.999)
        npt.assert_allclose(float(filt.weights[2]), 0.0, atol=ATOL)

    def test_confidence_to_noise_std_maps_confidence_to_scale(self):
        sigma = SO3ProductParticleFilter.confidence_to_noise_std(
            array([1.0, 0.5, 0.0]),
            noise_std=0.1,
            max_noise_std=1.0,
        )

        npt.assert_allclose(float(sigma[0]), 0.1, atol=ATOL)
        npt.assert_allclose(float(sigma[2]), 1.0, atol=ATOL)
        self.assertGreater(float(sigma[1]), 0.1)
        self.assertLess(float(sigma[1]), 1.0)

    def test_geodesic_log_likelihood_uses_confidence_and_component_noise(self):
        measurement = array([[0.0, 0.0, 0.0, 1.0], z_quaternion(pi / 2.0)])
        particles = stack(
            [
                array([[0.0, 0.0, 0.0, 1.0], z_quaternion(0.0)]),
                array([z_quaternion(pi / 2.0), z_quaternion(pi / 2.0)]),
            ],
            axis=0,
        )

        confidence_filter = SO3ProductParticleFilter(n_particles=2, num_rotations=2)
        confidence_filter.set_particles(particles)
        confidence_filter.update_with_geodesic_log_likelihood(
            measurement,
            noise_std=0.2,
            confidence=array([1.0, 0.0]),
            resample=False,
        )
        self.assertGreater(float(confidence_filter.weights[0]), 0.99)

        heteroskedastic_filter = SO3ProductParticleFilter(n_particles=2, num_rotations=2)
        heteroskedastic_filter.set_particles(particles)
        heteroskedastic_filter.update_with_geodesic_log_likelihood(
            measurement,
            component_noise_std=array([10.0, 0.2]),
            resample=False,
        )
        self.assertGreater(
            float(heteroskedastic_filter.weights[1]),
            float(heteroskedastic_filter.weights[0]),
        )

    def test_systematic_resampling_resets_weights(self):
        random.seed(0)
        filt = SO3ProductParticleFilter(n_particles=3, num_rotations=1)
        filt.set_particles(
            array(
                [
                    [[0.0, 0.0, 0.0, 1.0]],
                    [z_quaternion(pi / 4.0)],
                    [z_quaternion(pi / 2.0)],
                ]
            ),
            weights=array([0.8, 0.1, 0.1]),
        )

        indices = filt.resample_systematic()

        self.assertEqual(indices.shape, (3,))
        self.assertEqual(filt.particles.shape, (3, 1, 4))
        npt.assert_allclose(filt.weights, ones(3) / 3)
        npt.assert_allclose(linalg.norm(filt.particles, axis=-1), ones((3, 1)))


if __name__ == "__main__":
    unittest.main()
