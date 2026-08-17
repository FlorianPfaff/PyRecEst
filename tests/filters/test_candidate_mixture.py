import unittest

import numpy as np
from pyrecest.filters.candidate_mixture import GaussianMixtureMeasurementFactor


class GaussianMixtureMeasurementFactorTest(unittest.TestCase):
    def test_squared_isotropic_responsibilities_match_gaussian_potentials(self):
        factor = GaussianMixtureMeasurementFactor(
            means=np.array([[0.0], [2.0]]),
            covariances=np.array([[[1.0]], [[4.0]]]),
            log_weights=np.log(np.array([0.25, 0.75])),
        )

        result = factor.evaluate(np.array([1.0]))

        raw = np.array(
            [
                0.25 * np.exp(-0.5) / 1.0,
                0.75 * np.exp(-0.125) / 2.0,
            ]
        )
        expected = raw / raw.sum()
        np.testing.assert_allclose(result.responsibilities, expected)
        self.assertEqual(result.dominant_index, int(np.argmax(expected)))

    def test_huber_loss_robustifies_mahalanobis_distance(self):
        squared = GaussianMixtureMeasurementFactor(
            means=np.array([[0.0], [10.0]]),
            covariances=np.eye(1),
            loss="squared",
        ).evaluate(np.array([0.0]))
        huber = GaussianMixtureMeasurementFactor(
            means=np.array([[0.0], [10.0]]),
            covariances=np.eye(1),
            loss="huber",
            huber_delta=1.0,
        ).evaluate(np.array([0.0]))

        self.assertEqual(squared.robust_costs[1], 50.0)
        self.assertEqual(huber.robust_costs[1], 9.5)
        self.assertGreater(huber.responsibilities[1], squared.responsibilities[1])

    def test_observation_matrix_and_offset_map_state_to_measurement_space(self):
        factor = GaussianMixtureMeasurementFactor(
            means=np.array([[5.0, -1.0]]),
            covariances=np.eye(2),
            observation_matrix=np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 1.0]]),
            offset=np.array([1.0, -2.0]),
        )

        result = factor.evaluate(np.array([2.0, 1.0, 1.0]))

        np.testing.assert_allclose(result.predicted_measurement, np.array([5.0, -1.0]))
        np.testing.assert_allclose(result.residuals, np.zeros((1, 2)))
        np.testing.assert_allclose(result.responsibilities, np.ones(1))

    def test_moment_matching_includes_component_noise_and_between_mean_spread(self):
        factor = GaussianMixtureMeasurementFactor(
            means=np.array([[0.0], [2.0]]),
            covariances=np.array([[[1.0]], [[3.0]]]),
        )

        mean, covariance = factor.moment_match(np.array([0.25, 0.75]))

        np.testing.assert_allclose(mean, np.array([1.5]))
        np.testing.assert_allclose(covariance, np.array([[3.25]]))

    def test_log_determinant_weight_can_reproduce_custom_scale_penalty(self):
        factor = GaussianMixtureMeasurementFactor(
            means=np.zeros((2, 3)),
            covariances=np.array([np.eye(3), 4.0 * np.eye(3)]),
            log_determinant_weight=0.5,
        )

        result = factor.evaluate(np.zeros(3))

        # -0.5 * 0.5 * log(det(4 I_3)) = -1.5 log(2)
        self.assertAlmostEqual(
            result.component_log_weights[1] - result.component_log_weights[0],
            -1.5 * np.log(2.0),
        )

    def test_positive_infinite_prior_weights_share_responsibility(self):
        result = GaussianMixtureMeasurementFactor(
            means=np.array([[0.0], [1.0], [2.0]]),
            covariances=np.eye(1),
            log_weights=np.array([np.inf, 0.0, np.inf]),
        ).evaluate(np.array([1.0]))

        np.testing.assert_allclose(result.responsibilities, np.array([0.5, 0.0, 0.5]))
        self.assertTrue(np.isinf(result.log_evidence))

    def test_extreme_finite_log_weights_are_normalized_stably(self):
        result = GaussianMixtureMeasurementFactor(
            means=np.array([[0.0], [0.0]]),
            covariances=np.eye(1),
            log_weights=np.array([1.0e300, 1.0e300]),
        ).evaluate(np.array([0.0]))

        np.testing.assert_allclose(result.responsibilities, np.array([0.5, 0.5]))
        self.assertTrue(np.isfinite(result.log_evidence))

    def test_invalid_shapes_and_covariances_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "means"):
            GaussianMixtureMeasurementFactor(np.array([0.0, 1.0]), np.eye(1))
        with self.assertRaisesRegex(ValueError, "positive definite"):
            GaussianMixtureMeasurementFactor(
                np.array([[0.0, 0.0]]),
                np.array([[1.0, 0.0], [0.0, 0.0]]),
            )
        with self.assertRaisesRegex(ValueError, "log_weights"):
            GaussianMixtureMeasurementFactor(
                np.array([[0.0], [1.0]]),
                np.eye(1),
                log_weights=np.array([0.0]),
            )
        with self.assertRaisesRegex(ValueError, "state"):
            GaussianMixtureMeasurementFactor(
                np.array([[0.0, 0.0]]),
                np.eye(2),
            ).evaluate(np.array([0.0]))

    def test_bool_text_complex_and_temporal_inputs_are_rejected(self):
        invalid_means = (
            np.array([[True]]),
            np.array([["0.0"]]),
            np.array([[1.0 + 2.0j]]),
            np.array([[np.datetime64("2026-01-01")]]),
        )
        for means in invalid_means:
            with self.subTest(means=means):
                with self.assertRaisesRegex(ValueError, "means"):
                    GaussianMixtureMeasurementFactor(means, np.eye(1))


if __name__ == "__main__":
    unittest.main()
