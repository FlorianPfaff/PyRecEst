import math
import unittest

import numpy as np
from pyrecest.filters import OnlineTimeOffsetEstimator


class OnlineTimeOffsetEstimatorTest(unittest.TestCase):
    def test_update_from_position_residual_moves_offset_toward_projection(self):
        estimator = OnlineTimeOffsetEstimator(offset=0.0, variance=1.0)

        nis = estimator.update_from_position_residual(
            residual=np.array([10.0, 0.0]),
            velocity=np.array([5.0, 0.0]),
            measurement_variance=1.0,
        )

        self.assertTrue(math.isfinite(nis))
        self.assertGreater(estimator.offset, 0.0)
        self.assertLess(estimator.offset, 2.0)
        self.assertLess(estimator.variance, 1.0)

    def test_low_speed_returns_nan_without_update(self):
        estimator = OnlineTimeOffsetEstimator(offset=1.0, variance=2.0, min_speed=10.0)

        nis = estimator.update_from_position_residual(
            residual=np.array([10.0]),
            velocity=np.array([1.0]),
            measurement_variance=1.0,
        )

        self.assertTrue(math.isnan(nis))
        self.assertEqual(estimator.offset, 1.0)
        self.assertEqual(estimator.variance, 2.0)

    def test_predict_adds_process_variance(self):
        estimator = OnlineTimeOffsetEstimator(variance=1.0, process_variance=0.25)

        estimator.predict(dt=3.0)

        self.assertAlmostEqual(estimator.variance, 1.25)

    def test_constructor_rejects_nonfinite_scalar_controls(self):
        invalid_values = (np.nan, np.inf, -np.inf, True, np.array([1.0]))
        for field_name in ("offset", "variance", "process_variance", "min_speed"):
            for value in invalid_values:
                with self.subTest(field_name=field_name, value=value):
                    with self.assertRaisesRegex(
                        ValueError,
                        f"{field_name} must be a finite scalar",
                    ):
                        OnlineTimeOffsetEstimator(**{field_name: value})

    def test_update_rejects_invalid_measurement_inputs_without_state_change(self):
        invalid_updates = (
            {"measurement_variance": np.nan},
            {"measurement_variance": np.inf},
            {"measurement_variance": True},
            {"measurement_variance": np.array([1.0])},
            {"measurement_variance": -1.0},
            {"residual": np.array([np.nan])},
            {"velocity": np.array([np.inf])},
        )

        for override in invalid_updates:
            estimator = OnlineTimeOffsetEstimator(offset=1.0, variance=2.0)
            kwargs = {
                "residual": np.array([1.0]),
                "velocity": np.array([2.0]),
                "measurement_variance": 1.0,
            }
            kwargs.update(override)
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    estimator.update_from_position_residual(**kwargs)
                self.assertEqual(estimator.offset, 1.0)
                self.assertEqual(estimator.variance, 2.0)


if __name__ == "__main__":
    unittest.main()
