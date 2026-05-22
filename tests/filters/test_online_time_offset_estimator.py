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


if __name__ == "__main__":
    unittest.main()
