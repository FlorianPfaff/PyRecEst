import math
import unittest

import numpy as np
from pyrecest.filters import OnlineTimeOffsetEstimator


class OnlineTimeOffsetEstimatorMaskedInputsTest(unittest.TestCase):
    def test_constructor_rejects_masked_scalar_controls(self):
        for field_name in ("offset", "variance", "process_variance", "min_speed"):
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(
                    ValueError, f"{field_name} must be a finite scalar"
                ):
                    OnlineTimeOffsetEstimator(
                        **{field_name: np.ma.array(1.0, mask=True)}
                    )

    def test_predict_rejects_masked_dt_without_state_change(self):
        estimator = OnlineTimeOffsetEstimator(offset=1.0, variance=2.0)

        with self.assertRaisesRegex(ValueError, "dt must be a finite scalar"):
            estimator.predict(dt=np.ma.array(1.0, mask=True))

        self.assertEqual(estimator.offset, 1.0)
        self.assertEqual(estimator.variance, 2.0)

    def test_update_rejects_masked_inputs_without_state_change(self):
        invalid_updates = (
            {"residual": np.ma.array([1.0], mask=[True])},
            {"velocity": np.ma.array([2.0], mask=[True])},
            {"measurement_variance": np.ma.array(1.0, mask=True)},
            {"residual": [np.ma.masked]},
            {"velocity": np.array([np.ma.masked], dtype=object)},
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

    def test_clear_mask_wrappers_remain_supported(self):
        estimator = OnlineTimeOffsetEstimator(
            offset=np.ma.array(0.0, mask=False),
            variance=np.ma.array(1.0, mask=False),
            process_variance=np.ma.array(0.0, mask=False),
            min_speed=np.ma.array(0.0, mask=False),
        )

        estimator.predict(dt=np.ma.array(1.0, mask=False))
        nis = estimator.update_from_position_residual(
            residual=np.ma.array([2.0], mask=[False]),
            velocity=np.ma.array([2.0], mask=[False]),
            measurement_variance=np.ma.array(1.0, mask=False),
        )

        self.assertTrue(math.isfinite(nis))
        self.assertAlmostEqual(estimator.offset, 0.8)
        self.assertAlmostEqual(estimator.variance, 0.2)


if __name__ == "__main__":
    unittest.main()
