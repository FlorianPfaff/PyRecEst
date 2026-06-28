import unittest

import pyrecest.backend
from pyrecest.backend import array, eye, zeros
from pyrecest.filters import (
    normalize_active_measurement_mask,
    normalize_measurement_noise_covariances,
    normalize_measurement_reliability,
    normalize_measurement_weights,
)


def _as_covariance_matrix(value, dim, name):
    matrix = array(value)
    if matrix.ndim == 0:
        matrix = matrix * eye(dim)
    if matrix.ndim == 1:
        if matrix.shape[0] != dim:
            raise ValueError(f"{name} vector must have length {dim}")
        matrix = matrix * eye(dim)
    if matrix.shape != (dim, dim):
        raise ValueError(f"{name} must have shape ({dim}, {dim})")
    return matrix


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="measurement reliability tests currently use backend array shape checks",
)
class TestMeasurementReliability(unittest.TestCase):
    def test_scalar_weight_expands_to_all_measurements(self):
        weights = normalize_measurement_weights(0.25, 3)

        self.assertEqual(weights.shape, (3,))
        self.assertEqual([float(value) for value in weights], [0.25, 0.25, 0.25])

    def test_weight_vector_is_validated(self):
        weights = normalize_measurement_weights(array([1.0, 0.5, 0.0]), 3)

        self.assertEqual([float(value) for value in weights], [1.0, 0.5, 0.0])
        with self.assertRaises(ValueError):
            normalize_measurement_weights(array([1.0, 0.5]), 3)
        with self.assertRaises(ValueError):
            normalize_measurement_weights(array([1.0, -0.1, 0.0]), 3)

    def test_weight_inputs_must_be_real_numeric(self):
        invalid_weights = (
            True,
            array([True, False]),
            "0.25",
            array(["0.5", "1.0"]),
            1.0 + 0.0j,
            array([1.0 + 0.0j]),
        )

        for invalid_weight in invalid_weights:
            with self.subTest(weight=invalid_weight):
                with self.assertRaisesRegex(ValueError, "real numeric"):
                    normalize_measurement_weights(invalid_weight, 2)

    def test_scalar_mask_expands_to_all_measurements(self):
        self.assertEqual(
            normalize_active_measurement_mask(False, 3),
            [False, False, False],
        )
        self.assertEqual(
            normalize_active_measurement_mask(True, 2),
            [True, True],
        )

        for invalid_mask in ("False", 1):
            with self.subTest(mask=invalid_mask):
                with self.assertRaisesRegex(ValueError, "booleans"):
                    normalize_active_measurement_mask(invalid_mask, 2)

    def test_mask_vector_is_validated(self):
        self.assertEqual(
            normalize_active_measurement_mask(array([True, False, True]), 3),
            [True, False, True],
        )
        with self.assertRaises(ValueError):
            normalize_active_measurement_mask(array([True, False]), 3)
        with self.assertRaisesRegex(ValueError, "booleans"):
            normalize_active_measurement_mask(array([1, 0, 1]), 3)

    def test_reliability_selection_skips_masked_and_zero_weight_measurements(self):
        reliability = normalize_measurement_reliability(
            array([1.0, 0.0, 0.25, 1.0]),
            array([True, True, False, True]),
            4,
        )

        self.assertEqual(reliability.active_measurement_indices, [0, 3])
        self.assertEqual(reliability.active_measurement_mask, [True, True, False, True])

    def test_shared_covariance_is_stacked_per_measurement(self):
        covariances = normalize_measurement_noise_covariances(
            0.5,
            3,
            2,
            as_covariance_matrix=_as_covariance_matrix,
        )

        self.assertEqual(covariances.shape, (3, 2, 2))
        self.assertEqual(float(covariances[0, 0, 0]), 0.5)
        self.assertEqual(float(covariances[2, 1, 1]), 0.5)

    def test_empty_measurement_covariances_have_batch_shape(self):
        covariances = normalize_measurement_noise_covariances(
            0.5,
            0,
            2,
            as_covariance_matrix=_as_covariance_matrix,
        )

        self.assertEqual(covariances.shape, (0, 2, 2))

    def test_empty_measurement_covariances_validate_shared_noise(self):
        with self.assertRaisesRegex(ValueError, "R vector must have length 2"):
            normalize_measurement_noise_covariances(
                array([0.5, 0.25, 0.125]),
                0,
                2,
                as_covariance_matrix=_as_covariance_matrix,
            )

    def test_empty_per_measurement_covariances_have_batch_shape(self):
        covariances = normalize_measurement_noise_covariances(
            zeros((0, 2, 2)),
            0,
            2,
            as_covariance_matrix=_as_covariance_matrix,
        )

        self.assertEqual(covariances.shape, (0, 2, 2))

    def test_per_measurement_covariances_are_validated(self):
        covariances = normalize_measurement_noise_covariances(
            array([0.1 * eye(2), 0.2 * eye(2)]),
            2,
            2,
            as_covariance_matrix=_as_covariance_matrix,
        )

        self.assertEqual(covariances.shape, (2, 2, 2))
        self.assertEqual(float(covariances[0, 0, 0]), 0.1)
        self.assertEqual(float(covariances[1, 1, 1]), 0.2)
        with self.assertRaises(ValueError):
            normalize_measurement_noise_covariances(
                array([eye(2)]),
                2,
                2,
                as_covariance_matrix=_as_covariance_matrix,
            )


if __name__ == "__main__":
    unittest.main()
