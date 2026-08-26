import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.backend import array, to_numpy
from pyrecest.filters import KalmanFilter
from pyrecest.filters._linear_gaussian import (
    linear_gaussian_predict,
    linear_gaussian_update,
)


class LinearGaussianNonfiniteInputsTest(unittest.TestCase):
    @staticmethod
    def _state_snapshot(kf):
        state = kf.filter_state
        return (
            np.asarray(to_numpy(state.mu)).copy(),
            np.asarray(to_numpy(state.C)).copy(),
        )

    def test_predict_rejects_nonfinite_inputs(self):
        base = {
            "mean": array([0.0]),
            "covariance": array([[1.0]]),
            "system_matrix": array([[1.0]]),
            "sys_noise_cov": array([[0.1]]),
        }
        cases = (
            ("mean", array([float("nan")])),
            ("covariance", array([[float("inf")]])),
            ("system_matrix", array([[float("-inf")]])),
            ("sys_noise_cov", array([[float("nan")]])),
            ("sys_input", array([float("inf")])),
        )

        for name, value in cases:
            kwargs = dict(base)
            kwargs[name] = value
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError,
                f"{name} must contain only finite values",
            ):
                linear_gaussian_predict(**kwargs)

    def test_update_rejects_nonfinite_inputs(self):
        base = {
            "mean": array([0.0]),
            "covariance": array([[1.0]]),
            "measurement": array([0.0]),
            "measurement_matrix": array([[1.0]]),
            "meas_noise": array([[0.1]]),
        }
        cases = (
            ("mean", array([float("nan")])),
            ("covariance", array([[float("inf")]])),
            ("measurement", array([float("-inf")])),
            ("measurement_matrix", array([[float("nan")]])),
            ("meas_noise", array([[float("inf")]])),
        )

        for name, value in cases:
            kwargs = dict(base)
            kwargs[name] = value
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError,
                f"{name} must contain only finite values",
            ):
                linear_gaussian_update(**kwargs)

    def test_failed_predict_does_not_poison_kalman_state(self):
        kf = KalmanFilter((array([0.0]), array([[1.0]])))
        mean_before, covariance_before = self._state_snapshot(kf)

        with self.assertRaisesRegex(
            ValueError,
            "sys_noise_cov must contain only finite values",
        ):
            kf.predict_linear(array([[1.0]]), array([[float("nan")]]))

        mean_after, covariance_after = self._state_snapshot(kf)
        npt.assert_allclose(mean_after, mean_before)
        npt.assert_allclose(covariance_after, covariance_before)

    def test_failed_update_does_not_poison_kalman_state(self):
        kf = KalmanFilter((array([0.0]), array([[1.0]])))
        mean_before, covariance_before = self._state_snapshot(kf)

        with self.assertRaisesRegex(
            ValueError,
            "measurement must contain only finite values",
        ):
            kf.update_linear(
                array([float("inf")]),
                array([[1.0]]),
                array([[0.1]]),
            )

        mean_after, covariance_after = self._state_snapshot(kf)
        npt.assert_allclose(mean_after, mean_before)
        npt.assert_allclose(covariance_after, covariance_before)


if __name__ == "__main__":
    unittest.main()
