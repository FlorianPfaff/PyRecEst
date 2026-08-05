import unittest

import numpy as np
import numpy.testing as npt
from pyrecest.backend import __backend_name__, array, eye, to_numpy
from pyrecest.filters._linear_gaussian import (
    linear_gaussian_innovation,
    linear_gaussian_predict,
    linear_gaussian_update,
)


@unittest.skipUnless(
    __backend_name__ == "numpy",
    reason="strict floating-point overflow checks require the NumPy backend",
)
class LinearGaussianExtremeCovarianceTest(unittest.TestCase):
    @staticmethod
    def _covariance():
        return array([[1e308, 0.0], [0.0, 2e307]])

    def test_predict_does_not_overflow_while_symmetrizing_finite_covariance(self):
        covariance = self._covariance()

        with np.errstate(over="raise", invalid="raise"):
            predicted_mean, predicted_covariance = linear_gaussian_predict(
                array([0.0, 0.0]),
                covariance,
                eye(2),
                array([[0.0, 0.0], [0.0, 0.0]]),
            )

        npt.assert_array_equal(to_numpy(predicted_mean), np.zeros(2))
        npt.assert_array_equal(to_numpy(predicted_covariance), to_numpy(covariance))

    def test_innovation_does_not_overflow_while_symmetrizing_finite_covariance(self):
        covariance = self._covariance()

        with np.errstate(over="raise", invalid="raise"):
            innovation, innovation_covariance = linear_gaussian_innovation(
                array([0.0, 0.0]),
                covariance,
                array([0.0, 0.0]),
                eye(2),
                array([[0.0, 0.0], [0.0, 0.0]]),
            )

        npt.assert_array_equal(to_numpy(innovation), np.zeros(2))
        npt.assert_array_equal(to_numpy(innovation_covariance), to_numpy(covariance))

    def test_update_does_not_overflow_while_symmetrizing_finite_covariance(self):
        covariance = self._covariance()

        with np.errstate(over="raise", invalid="raise"):
            updated_mean, updated_covariance = linear_gaussian_update(
                array([0.0, 0.0]),
                covariance,
                array([0.0]),
                array([[0.0, 0.0]]),
                array([[1.0]]),
            )

        npt.assert_array_equal(to_numpy(updated_mean), np.zeros(2))
        npt.assert_array_equal(to_numpy(updated_covariance), to_numpy(covariance))


if __name__ == "__main__":
    unittest.main()
