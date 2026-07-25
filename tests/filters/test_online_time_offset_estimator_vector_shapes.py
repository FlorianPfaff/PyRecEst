import unittest

import numpy as np
from pyrecest.filters import OnlineTimeOffsetEstimator


class OnlineTimeOffsetEstimatorVectorShapeTest(unittest.TestCase):
    def test_update_rejects_matrix_vectors_without_state_change(self):
        matrix_inputs = (
            {"residual": np.array([[1.0, 2.0]])},
            {"residual": np.array([[1.0], [2.0]])},
            {"velocity": np.array([[2.0, 3.0]])},
            {"velocity": np.array([[2.0], [3.0]])},
        )

        for override in matrix_inputs:
            estimator = OnlineTimeOffsetEstimator(offset=1.0, variance=2.0)
            kwargs = {
                "residual": np.array([1.0, 2.0]),
                "velocity": np.array([2.0, 3.0]),
                "measurement_variance": 1.0,
            }
            kwargs.update(override)

            with self.subTest(override=override):
                with self.assertRaisesRegex(
                    ValueError,
                    "(residual|velocity) must be one-dimensional",
                ):
                    estimator.update_from_position_residual(**kwargs)
                self.assertEqual(estimator.offset, 1.0)
                self.assertEqual(estimator.variance, 2.0)

    def test_update_preserves_scalar_one_dimensional_inputs(self):
        estimator = OnlineTimeOffsetEstimator(offset=0.0, variance=1.0)

        nis = estimator.update_from_position_residual(
            residual=10.0,
            velocity=5.0,
            measurement_variance=1.0,
        )

        self.assertTrue(np.isfinite(nis))
        self.assertGreater(estimator.offset, 0.0)
        self.assertLess(estimator.offset, 2.0)


if __name__ == "__main__":
    unittest.main()
