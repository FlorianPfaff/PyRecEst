import unittest

import numpy as np
from pyrecest.backend import array, eye, to_numpy
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
    if matrix.shape != (dim, dim):
        raise ValueError(f"{name} must have shape ({dim}, {dim})")
    return matrix


class TestMeasurementReliabilityMaskedInputs(unittest.TestCase):
    def test_masked_counts_are_rejected_before_payload_conversion(self):
        masked_count = np.ma.array(2, mask=True)
        with self.assertRaisesRegex(ValueError, "n_measurements"):
            normalize_measurement_weights(None, masked_count)
        with self.assertRaisesRegex(ValueError, "n_measurements"):
            normalize_measurement_reliability(None, None, masked_count)

        masked_dim = np.ma.array(1, mask=True)
        with self.assertRaisesRegex(ValueError, "measurement_dim"):
            normalize_measurement_noise_covariances(
                1.0,
                1,
                masked_dim,
                as_covariance_matrix=_as_covariance_matrix,
            )

    def test_masked_measurement_weights_are_rejected(self):
        invalid_weights = (
            np.ma.array(0.5, mask=True),
            np.ma.array([1.0, 0.5], mask=[False, True]),
            [1.0, np.ma.array(0.5, mask=True)],
        )
        for weights in invalid_weights:
            with self.subTest(weights=weights):
                with self.assertRaisesRegex(ValueError, "masked values"):
                    normalize_measurement_weights(weights, 2)

    def test_masked_active_measurement_flags_are_rejected(self):
        invalid_masks = (
            np.ma.array(True, mask=True),
            np.ma.array([True, False], mask=[False, True]),
            [True, np.ma.array(False, mask=True)],
        )
        for active_mask in invalid_masks:
            with self.subTest(active_mask=active_mask):
                with self.assertRaisesRegex(ValueError, "masked values"):
                    normalize_active_measurement_mask(active_mask, 2)

    def test_masked_measurement_noise_is_rejected(self):
        shared_noise = np.ma.array([[1.0]], mask=[[True]])
        with self.assertRaisesRegex(ValueError, "R must not contain masked values"):
            normalize_measurement_noise_covariances(
                shared_noise,
                1,
                1,
                as_covariance_matrix=_as_covariance_matrix,
            )

        batched_noise = np.ma.array(
            [[[1.0]], [[2.0]]],
            mask=[[[False]], [[True]]],
        )
        with self.assertRaisesRegex(ValueError, "noise must not contain masked values"):
            normalize_measurement_noise_covariances(
                batched_noise,
                2,
                1,
                as_covariance_matrix=_as_covariance_matrix,
                name="noise",
            )

    def test_fully_unmasked_masked_arrays_remain_supported(self):
        weights = normalize_measurement_weights(
            np.ma.array([1.0, 0.5], mask=False),
            2,
        )
        np.testing.assert_allclose(to_numpy(weights), np.array([1.0, 0.5]))

        active_mask = normalize_active_measurement_mask(
            np.ma.array([True, False], mask=False),
            2,
        )
        self.assertEqual(active_mask, [True, False])

        noise = normalize_measurement_noise_covariances(
            np.ma.array([[2.0]], mask=False),
            2,
            1,
            as_covariance_matrix=_as_covariance_matrix,
        )
        np.testing.assert_allclose(to_numpy(noise), np.array([[[2.0]], [[2.0]]]))


if __name__ == "__main__":
    unittest.main()
