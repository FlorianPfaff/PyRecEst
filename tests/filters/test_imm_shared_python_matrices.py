import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, eye, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.interacting_multiple_model_filter import (
    InteractingMultipleModelFilter,
)


class _LinearGaussianFilter:
    def __init__(self, mean):
        mean = array(mean)
        self.filter_state = GaussianDistribution(
            mean, eye(mean.shape[0]), check_validity=False
        )

    def predict_linear(self, system_matrix, sys_noise_cov, sys_input=None):
        mean = system_matrix @ self.filter_state.mu
        if sys_input is not None:
            mean = mean + sys_input
        covariance = (
            system_matrix @ self.filter_state.C @ system_matrix.T + sys_noise_cov
        )
        self.filter_state = GaussianDistribution(
            mean, covariance, check_validity=False
        )


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Only supported on numpy backend",
)
class IMMSharedPythonMatrixTest(unittest.TestCase):
    def test_predict_linear_broadcasts_shared_python_matrix(self):
        imm = InteractingMultipleModelFilter(
            [_LinearGaussianFilter([1.0, 2.0]), _LinearGaussianFilter([3.0, 4.0])],
            transition_matrix=eye(2),
            mode_probabilities=array([0.5, 0.5]),
        )

        imm.predict_linear(
            [[1.0, 1.0], [0.0, 1.0]],
            zeros((2, 2)),
        )

        npt.assert_allclose(imm.filter_bank[0].filter_state.mu, array([3.0, 2.0]))
        npt.assert_allclose(imm.filter_bank[1].filter_state.mu, array([7.0, 4.0]))
        expected_covariance = array([[2.0, 1.0], [1.0, 1.0]])
        npt.assert_allclose(imm.filter_bank[0].filter_state.C, expected_covariance)
        npt.assert_allclose(imm.filter_bank[1].filter_state.C, expected_covariance)


if __name__ == "__main__":
    unittest.main()
