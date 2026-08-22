import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, diag, to_numpy
from pyrecest.distributions import LinearDiracDistribution
from pyrecest.filters import EDHParticleFlowFilter, LEDHParticleFlowFilter
from pyrecest.filters.daum_huang_particle_filter import (
    gaussian_bridge_moments,
    gaussian_flow_affine_increment,
    ledh_particle_flow,
)


def _mean_and_cov(particles):
    distribution = LinearDiracDistribution(array(particles))
    return distribution.mean(), distribution.covariance()


class SquareMeasurementModel:
    noise_covariance = array([[0.2]])

    def measurement_function(self, state):
        return array([state[0] * state[0]])

    def jacobian(self, state):
        return array([[2.0 * state[0]]])


class DaumHuangParticleFlowFilterTest(unittest.TestCase):
    def test_affine_increment_matches_bridge_moments(self):
        particles = array(
            [
                [-2.0, -1.0],
                [-1.0, 1.0],
                [0.0, 0.0],
                [1.0, -1.0],
                [2.0, 1.0],
            ]
        )
        mean, covariance = _mean_and_cov(particles)
        measurement_matrix = array([[1.0, -0.5]])
        measurement = array([0.25])
        measurement_noise = array([[0.75]])

        transported = gaussian_flow_affine_increment(
            particles,
            mean,
            covariance,
            measurement_matrix,
            measurement,
            measurement_noise,
            1.0,
            jitter=0.0,
        )
        expected_mean, expected_covariance = gaussian_bridge_moments(
            mean,
            covariance,
            measurement_matrix,
            measurement,
            measurement_noise,
            1.0,
            jitter=0.0,
        )
        actual_mean, actual_covariance = _mean_and_cov(transported)

        npt.assert_allclose(to_numpy(actual_mean), to_numpy(expected_mean), atol=1e-10)
        npt.assert_allclose(
            to_numpy(actual_covariance), to_numpy(expected_covariance), atol=1e-10
        )

    def test_bridge_rejects_nonsymmetric_covariances(self):
        mean = array([0.0, 0.0])
        identity_covariance = array([[1.0, 0.0], [0.0, 1.0]])
        nonsymmetric_covariance = array([[1.0, 0.5], [0.0, 1.0]])

        with self.assertRaisesRegex(ValueError, "symmetric"):
            gaussian_bridge_moments(
                mean,
                nonsymmetric_covariance,
                array([[1.0, 0.0]]),
                array([0.0]),
                array([[1.0]]),
                1.0,
                jitter=0.0,
            )

        with self.assertRaisesRegex(ValueError, "symmetric"):
            gaussian_bridge_moments(
                mean,
                identity_covariance,
                identity_covariance,
                array([0.0, 0.0]),
                nonsymmetric_covariance,
                1.0,
                jitter=0.0,
            )

    def test_bridge_rejects_complex_numpy_inputs_without_truncation(self):
        real_vector = array([0.0])
        real_matrix = array([[1.0]])

        invalid_cases = [
            (np.array([1.0 + 2.0j]), real_matrix),
            (real_vector, np.array([[1.0 + 2.0j]])),
        ]
        for mean, covariance in invalid_cases:
            with self.subTest(mean=mean, covariance=covariance):
                with self.assertRaisesRegex(ValueError, "real-valued"):
                    gaussian_bridge_moments(
                        mean,
                        covariance,
                        real_matrix,
                        real_vector,
                        real_matrix,
                        1.0,
                        jitter=0.0,
                    )

    def test_bridge_exponent_must_be_finite_nonnegative_scalar(self):
        mean = array([0.0])
        covariance = array([[1.0]])
        measurement_matrix = array([[1.0]])
        measurement = array([0.0])
        measurement_noise = array([[1.0]])

        for delta_lambda in (float("nan"), float("inf"), -0.1, True, [1.0]):
            with self.subTest(delta_lambda=delta_lambda):
                with self.assertRaisesRegex(ValueError, "delta_lambda"):
                    gaussian_bridge_moments(
                        mean,
                        covariance,
                        measurement_matrix,
                        measurement,
                        measurement_noise,
                        delta_lambda,
                    )

        with self.assertRaisesRegex(ValueError, "delta_lambda"):
            gaussian_flow_affine_increment(
                array([[0.0], [1.0]]),
                mean,
                covariance,
                measurement_matrix,
                measurement,
                measurement_noise,
                float("nan"),
            )

    def test_edh_filter_linear_update_matches_bridge_moments(self):
        particles = array(
            [
                [-2.0, -1.0],
                [-1.0, 1.0],
                [0.0, 0.0],
                [1.0, -1.0],
                [2.0, 1.0],
            ]
        )
        prior_mean, prior_covariance = _mean_and_cov(particles)
        measurement_matrix = array([[1.0, -0.5]])
        measurement = array([0.25])
        measurement_noise = array([[0.75]])
        expected_mean, expected_covariance = gaussian_bridge_moments(
            prior_mean,
            prior_covariance,
            measurement_matrix,
            measurement,
            measurement_noise,
            1.0,
            jitter=0.0,
        )
        filt = EDHParticleFlowFilter(
            n_particles=particles.shape[0],
            dim=particles.shape[1],
            n_steps=4,
            jitter=0.0,
        )
        filt.filter_state = LinearDiracDistribution(particles)

        info = filt.update_linear(
            measurement,
            measurement_matrix,
            measurement_noise,
            return_info=True,
        )
        actual_mean = filt.filter_state.mean()
        actual_covariance = filt.filter_state.covariance()

        self.assertEqual(info.flow_type, "edh")
        self.assertEqual(info.n_steps, 4)
        npt.assert_allclose(
            to_numpy(filt.filter_state.w),
            np.full(particles.shape[0], 1.0 / particles.shape[0]),
        )
        npt.assert_allclose(to_numpy(actual_mean), to_numpy(expected_mean), atol=1e-10)
        npt.assert_allclose(
            to_numpy(actual_covariance), to_numpy(expected_covariance), atol=1e-10
        )

    def test_particle_flow_filters_preserve_nonuniform_weights(self):
        particles = array(
            [
                [-2.0, -1.0],
                [-1.0, 1.0],
                [0.0, 0.0],
                [1.0, -1.0],
                [2.0, 1.0],
            ]
        )
        weights = array([0.05, 0.10, 0.15, 0.25, 0.45])
        prior = LinearDiracDistribution(particles, weights)
        expected_weights = to_numpy(prior.w).copy()
        measurement_matrix = array([[1.0, -0.5]])
        measurement = array([0.25])
        measurement_noise = array([[0.75]])
        expected_mean, expected_covariance = gaussian_bridge_moments(
            prior.mean(),
            prior.covariance(),
            measurement_matrix,
            measurement,
            measurement_noise,
            1.0,
            jitter=0.0,
        )

        for filter_type in (EDHParticleFlowFilter, LEDHParticleFlowFilter):
            with self.subTest(filter_type=filter_type.__name__):
                filt = filter_type(
                    n_particles=particles.shape[0],
                    dim=particles.shape[1],
                    n_steps=4,
                    jitter=0.0,
                )
                filt.filter_state = LinearDiracDistribution(particles, weights)

                filt.update_linear(measurement, measurement_matrix, measurement_noise)

                npt.assert_allclose(to_numpy(filt.filter_state.w), expected_weights)
                npt.assert_allclose(
                    to_numpy(filt.filter_state.mean()),
                    to_numpy(expected_mean),
                    atol=1e-10,
                )
                npt.assert_allclose(
                    to_numpy(filt.filter_state.covariance()),
                    to_numpy(expected_covariance),
                    atol=1e-10,
                )

    def test_ledh_records_particlewise_linearization_points(self):
        particles = array([[-2.0], [-1.0], [1.0], [2.0]])

        transported, info = ledh_particle_flow(
            particles,
            SquareMeasurementModel(),
            array([1.0]),
            n_steps=2,
            return_info=True,
        )

        self.assertEqual(info.flow_type, "ledh")
        self.assertEqual(info.n_steps, 2)
        self.assertEqual(to_numpy(info.linearization_points[0]).shape, (4, 1))
        self.assertEqual(to_numpy(transported).shape, (4, 1))
        self.assertTrue(np.all(np.isfinite(to_numpy(transported))))

    def test_ledh_filter_nonlinear_update_accepts_scalar_callbacks(self):
        particles = array([[-2.0], [-1.0], [1.0], [2.0]])
        filt = LEDHParticleFlowFilter(n_particles=particles.shape[0], dim=1, n_steps=2)
        filt.filter_state = LinearDiracDistribution(particles)

        info = filt.update_nonlinear(
            array([1.0]),
            lambda state: array([state[0] * state[0]]),
            diag(array([0.2])),
            lambda state: array([[2.0 * state[0]]]),
            return_info=True,
        )

        self.assertEqual(info.flow_type, "ledh")
        self.assertTrue(np.all(np.isfinite(to_numpy(filt.filter_state.d))))

    def test_update_model_requires_jacobian(self):
        filt = EDHParticleFlowFilter(n_particles=2, dim=1)

        class MissingJacobian:
            noise_covariance = array([[1.0]])

            def measurement_function(self, state):
                return array([state[0]])

        with self.assertRaisesRegex(TypeError, "jacobian"):
            filt.update_model(MissingJacobian(), array([0.0]))


if __name__ == "__main__":
    unittest.main()
