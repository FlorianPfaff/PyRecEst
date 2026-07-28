import unittest

import numpy as np
from pyrecest.filters import (
    normalize_active_measurement_mask,
    normalize_measurement_noise_covariances,
    normalize_measurement_weights,
)


class TestMeasurementReliabilityTemporalCounts(unittest.TestCase):
    def test_measurement_count_rejects_temporal_scalars(self):
        invalid_counts = (
            np.timedelta64(2, "ns"),
            np.datetime64(2, "ns"),
            np.array(np.timedelta64(2, "ns"), dtype=object),
            np.array(np.datetime64(2, "ns"), dtype=object),
        )

        for value in invalid_counts:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    normalize_measurement_weights(None, value)
                with self.assertRaisesRegex(ValueError, "non-negative integer"):
                    normalize_active_measurement_mask(None, value)

    def test_measurement_dimension_rejects_temporal_scalars(self):
        invalid_dimensions = (
            np.timedelta64(1, "ns"),
            np.datetime64(1, "ns"),
            np.array(np.timedelta64(1, "ns"), dtype=object),
            np.array(np.datetime64(1, "ns"), dtype=object),
        )

        for value in invalid_dimensions:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    normalize_measurement_noise_covariances(
                        1.0,
                        1,
                        value,
                        as_covariance_matrix=lambda *_args: None,
                    )

    def test_numpy_integer_counts_remain_supported(self):
        weights = normalize_measurement_weights(None, np.int64(2))
        self.assertEqual(weights.shape, (2,))


if __name__ == "__main__":
    unittest.main()
