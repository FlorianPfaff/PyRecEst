import unittest

import numpy as np

from pyrecest.backend import array
from pyrecest.smoothers import (
    FactorizedGIWRandomMatrixTrackerState,
    FixedLagFactorizedGIWRandomMatrixSmoother,
    FixedLagRandomMatrixSmoother,
    RandomMatrixTrackerState,
)


class FixedLagRandomMatrixLagValidationTest(unittest.TestCase):
    @staticmethod
    def _smoother_state_cases():
        return (
            (
                FixedLagRandomMatrixSmoother,
                RandomMatrixTrackerState(
                    array([0.0]),
                    array([[1.0]]),
                    array([[1.0]]),
                    1.0,
                ),
            ),
            (
                FixedLagFactorizedGIWRandomMatrixSmoother,
                FactorizedGIWRandomMatrixTrackerState(
                    array([0.0]),
                    array([[1.0]]),
                    8.0,
                    array([[4.0]]),
                ),
            ),
        )

    def test_constructors_reject_lossy_lag_coercions(self):
        invalid_lags = (
            1.5,
            0.5,
            True,
            False,
            np.bool_(True),
            np.array(True),
            np.ma.array(2, mask=True),
        )

        for smoother_type, _ in self._smoother_state_cases():
            for lag in invalid_lags:
                with self.subTest(smoother=smoother_type.__name__, lag=repr(lag)):
                    with self.assertRaisesRegex(
                        ValueError,
                        "lag must be a non-negative integer",
                    ):
                        smoother_type(lag=lag)

    def test_smooth_overrides_reject_lossy_lag_coercions(self):
        for smoother_type, state in self._smoother_state_cases():
            smoother = smoother_type(lag=0)
            for lag in (1.5, 0.5, True, np.ma.array(2, mask=True)):
                with self.subTest(smoother=smoother_type.__name__, lag=repr(lag)):
                    with self.assertRaisesRegex(
                        ValueError,
                        "lag must be a non-negative integer",
                    ):
                        smoother.smooth([state], lag=lag)

    def test_exact_numpy_integer_lags_remain_supported(self):
        for smoother_type, state in self._smoother_state_cases():
            smoother = smoother_type(lag=np.int64(0))
            smoothed_states, smoother_gains = smoother.smooth(
                [state],
                lag=np.array(0, dtype=np.int64),
            )

            self.assertEqual(smoother.lag, 0)
            self.assertEqual(len(smoothed_states), 1)
            self.assertEqual(smoother_gains, [[]])

    def test_clear_mask_integer_lags_remain_supported(self):
        clear_mask_lag = np.ma.array(0, mask=False)
        for smoother_type, state in self._smoother_state_cases():
            smoother = smoother_type(lag=clear_mask_lag)
            smoothed_states, _ = smoother.smooth([state], lag=clear_mask_lag)

            self.assertEqual(smoother.lag, 0)
            self.assertEqual(len(smoothed_states), 1)


if __name__ == "__main__":
    unittest.main()
