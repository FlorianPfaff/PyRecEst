import unittest

from pyrecest.backend import array, eye, zeros
from pyrecest.models.validation import (
    infer_state_dim_from_distribution,
    validate_covariance_matrix,
    validate_measurement_matrix,
    validate_measurement_vector,
    validate_noise_covariance,
    validate_state_vector,
    validate_transition_matrix,
)


class TestModelValidation(unittest.TestCase):
    def test_validate_state_vector_accepts_expected_shape(self):
        state = validate_state_vector(array([1.0, 2.0]), state_dim=2)

        self.assertEqual(state.shape, (2,))

    def test_validate_state_vector_rejects_matrix(self):
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            validate_state_vector(array([[1.0, 2.0]]), state_dim=2)

    def test_validate_state_vector_rejects_wrong_dimension(self):
        with self.assertRaisesRegex(ValueError, "expected 3"):
            validate_state_vector(array([1.0, 2.0]), state_dim=3)

    def test_validate_state_vector_can_accept_scalar_for_one_dimensional_state(self):
        state = validate_state_vector(array(1.0), state_dim=1, allow_scalar=True)

        self.assertEqual(state.shape, (1,))

    def test_validate_measurement_vector_accepts_expected_shape(self):
        measurement = validate_measurement_vector(array([1.0]), meas_dim=1)

        self.assertEqual(measurement.shape, (1,))

    def test_validate_covariance_matrix_accepts_square_matrix(self):
        covariance = validate_covariance_matrix(eye(2), dim=2)

        self.assertEqual(covariance.shape, (2, 2))

    def test_validate_covariance_matrix_rejects_non_square_matrix(self):
        with self.assertRaisesRegex(ValueError, "square"):
            validate_covariance_matrix(array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

    def test_validate_covariance_matrix_rejects_wrong_dimension(self):
        with self.assertRaisesRegex(ValueError, "expected 3"):
            validate_covariance_matrix(eye(2), dim=3)

    def test_validate_covariance_matrix_can_check_symmetry(self):
        with self.assertRaisesRegex(ValueError, "symmetric"):
            validate_covariance_matrix(
                array([[1.0, 2.0], [0.0, 1.0]]), check_symmetric=True
            )

    def test_validate_noise_covariance_uses_covariance_rules(self):
        noise_covariance = validate_noise_covariance(
            array(0.5), dim=1, allow_scalar=True
        )

        self.assertEqual(noise_covariance.shape, (1, 1))

    def test_validate_transition_matrix_accepts_pred_by_state_shape(self):
        system_matrix = validate_transition_matrix(
            zeros((3, 2)), state_dim=2, pred_dim=3
        )

        self.assertEqual(system_matrix.shape, (3, 2))

    def test_validate_transition_matrix_rejects_wrong_state_dimension(self):
        with self.assertRaisesRegex(ValueError, "expected 3"):
            validate_transition_matrix(zeros((2, 2)), state_dim=3)

    def test_validate_measurement_matrix_accepts_meas_by_state_shape(self):
        measurement_matrix = validate_measurement_matrix(
            zeros((1, 2)), state_dim=2, meas_dim=1
        )

        self.assertEqual(measurement_matrix.shape, (1, 2))

    def test_validate_measurement_matrix_rejects_wrong_measurement_dimension(self):
        with self.assertRaisesRegex(ValueError, "expected 2"):
            validate_measurement_matrix(zeros((1, 2)), meas_dim=2)

    def test_infer_state_dim_from_explicit_dim(self):
        class DistributionWithDim:
            dim = 4

        self.assertEqual(infer_state_dim_from_distribution(DistributionWithDim()), 4)

    def test_infer_state_dim_from_mean_attribute(self):
        class DistributionWithMean:
            mu = array([0.0, 1.0, 2.0])

        self.assertEqual(infer_state_dim_from_distribution(DistributionWithMean()), 3)

    def test_infer_state_dim_from_covariance_method(self):
        class DistributionWithCovariance:
            def covariance(self):
                return eye(5)

        self.assertEqual(
            infer_state_dim_from_distribution(DistributionWithCovariance()), 5
        )

    def test_infer_state_dim_from_dirac_locations(self):
        class DistributionWithDiracs:
            d = zeros((7, 3))

        self.assertEqual(infer_state_dim_from_distribution(DistributionWithDiracs()), 3)

    def test_infer_state_dim_raises_for_unknown_distribution_shape(self):
        with self.assertRaisesRegex(ValueError, "Could not infer"):
            infer_state_dim_from_distribution(object())


if __name__ == "__main__":
    unittest.main()
