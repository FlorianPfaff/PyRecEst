import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.backend import array, zeros
from pyrecest.filters._ukf import UnscentedKalmanFilter, _UKFModel
from pyrecest.sampling.sigma_points import MerweScaledSigmaPoints


class UKFNonfiniteAtomicityTest(unittest.TestCase):
    @staticmethod
    def _make_filter(*, fx=None, hx=None):
        points = MerweScaledSigmaPoints(n=1, alpha=1.0, beta=2.0, kappa=0.0)
        model = _UKFModel(
            dim_x=1,
            dim_z=1,
            dt=1.0,
            hx=(lambda x: x) if hx is None else hx,
            fx=(lambda x, _dt: x) if fx is None else fx,
            points=points,
        )
        ukf = UnscentedKalmanFilter(model)
        ukf.x = array([0.25])
        ukf.P = array([[0.5]])
        ukf.Q = zeros((1, 1))
        ukf.R = array([[0.1]])
        return ukf

    @staticmethod
    def _assert_state_unchanged(ukf, x_before, p_before):
        npt.assert_allclose(np.asarray(ukf.x), x_before)
        npt.assert_allclose(np.asarray(ukf.P), p_before)

    def test_predict_rejects_nonfinite_transition_without_state_change(self):
        ukf = self._make_filter(
            fx=lambda _x, _dt: array([float("nan")]),
        )
        x_before = np.asarray(ukf.x).copy()
        p_before = np.asarray(ukf.P).copy()

        with self.assertRaisesRegex(
            ValueError,
            "transition function output must contain only finite values",
        ):
            ukf.predict()

        self._assert_state_unchanged(ukf, x_before, p_before)

    def test_update_rejects_nonfinite_measurement_without_state_change(self):
        ukf = self._make_filter()
        x_before = np.asarray(ukf.x).copy()
        p_before = np.asarray(ukf.P).copy()

        with self.assertRaisesRegex(
            ValueError,
            "measurement z must contain only finite values",
        ):
            ukf.update(array([float("nan")]))

        self._assert_state_unchanged(ukf, x_before, p_before)

    def test_update_rejects_nonfinite_model_output_without_state_change(self):
        ukf = self._make_filter(hx=lambda _x: array([float("inf")]))
        x_before = np.asarray(ukf.x).copy()
        p_before = np.asarray(ukf.P).copy()

        with self.assertRaisesRegex(
            ValueError,
            "measurement function output must contain only finite values",
        ):
            ukf.update(array([0.0]))

        self._assert_state_unchanged(ukf, x_before, p_before)

    def test_failed_predicted_covariance_validation_is_atomic(self):
        ukf = self._make_filter()
        ukf.Q = array([[float("nan")]])
        x_before = np.asarray(ukf.x).copy()
        p_before = np.asarray(ukf.P).copy()

        with self.assertRaisesRegex(ValueError, "P must contain only finite values"):
            ukf.predict()

        self._assert_state_unchanged(ukf, x_before, p_before)


if __name__ == "__main__":
    unittest.main()
