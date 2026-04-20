import copy
import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import allclose, array, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.interacting_multiple_model_filter import (
    IMM,
    InteractingMultipleModelFilter,
)


class MockGaussianFilter:
    def __init__(self, initial_state: GaussianDistribution):
        self.filter_state = copy.deepcopy(initial_state)

    @property
    def dim(self):
        return self.filter_state.dim

    def predict_identity(self, sys_noise_cov, sys_input=None):
        self.predict_linear(eye(self.dim), sys_noise_cov, sys_input)

    def predict_linear(self, system_matrix, sys_noise_cov, sys_input=None):
        mu = system_matrix @ self.filter_state.mu
        if sys_input is not None:
            mu = mu + sys_input
        covariance = (
            system_matrix @ self.filter_state.C @ system_matrix.T + sys_noise_cov
        )
        self.filter_state = GaussianDistribution(mu, covariance, check_validity=False)

    def update_identity(self, measurement, meas_noise):
        self.update_linear(measurement, eye(self.dim), meas_noise)

    def update_linear(self, measurement, measurement_matrix, meas_noise):
        innovation = measurement - measurement_matrix @ self.filter_state.mu
        innovation_covariance = (
            measurement_matrix @ self.filter_state.C @ measurement_matrix.T + meas_noise
        )
        kalman_gain = (
            self.filter_state.C
            @ measurement_matrix.T
            @ pyrecest.backend.linalg.inv(innovation_covariance)
        )
        mu = self.filter_state.mu + kalman_gain @ innovation
        identity_matrix = eye(self.dim)
        covariance = (
            identity_matrix - kalman_gain @ measurement_matrix
        ) @ self.filter_state.C @ (
            identity_matrix - kalman_gain @ measurement_matrix
        ).T + kalman_gain @ meas_noise @ kalman_gain.T
        covariance = 0.5 * (covariance + covariance.T)
        self.filter_state = GaussianDistribution(mu, covariance, check_validity=False)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Only supported on numpy backend",
)
class InteractingMultipleModelFilterTest(unittest.TestCase):
    def test_alias_exists(self):
        self.assertIs(IMM, InteractingMultipleModelFilter)

    def test_combined_state_matches_weighted_gaussian_mixture(self):
        filter_bank = [
            MockGaussianFilter(GaussianDistribution(array([0.0]), array([[1.0]]))),
            MockGaussianFilter(GaussianDistribution(array([10.0]), array([[1.0]]))),
        ]
        imm = InteractingMultipleModelFilter(
            filter_bank,
            transition_matrix=array([[1.0, 0.0], [0.0, 1.0]]),
            mode_probabilities=array([0.25, 0.75]),
        )

        npt.assert_allclose(imm.get_point_estimate(), array([7.5]))
        npt.assert_allclose(imm.combined_filter_state.C, array([[19.75]]))

    def test_predict_linear_supports_model_specific_arguments(self):
        filter_bank = [
            MockGaussianFilter(GaussianDistribution(array([0.0]), array([[1.0]]))),
            MockGaussianFilter(GaussianDistribution(array([10.0]), array([[1.0]]))),
        ]
        imm = InteractingMultipleModelFilter(
            filter_bank,
            transition_matrix=array([[0.9, 0.1], [0.1, 0.9]]),
            mode_probabilities=array([0.5, 0.5]),
        )

        imm.predict_linear(
            [array([[1.0]]), array([[2.0]])],
            [array([[0.0]]), array([[0.0]])],
        )

        npt.assert_allclose(imm.mode_probabilities, array([0.5, 0.5]))
        npt.assert_allclose(imm.filter_bank[0].filter_state.mu, array([1.0]))
        npt.assert_allclose(imm.filter_bank[1].filter_state.mu, array([18.0]))
        npt.assert_allclose(imm.filter_bank[0].filter_state.C, array([[10.0]]))
        npt.assert_allclose(imm.filter_bank[1].filter_state.C, array([[40.0]]))
        npt.assert_allclose(imm.get_point_estimate(), array([9.5]))

    def test_update_linear_shifts_probability_to_matching_model(self):
        filter_bank = [
            MockGaussianFilter(GaussianDistribution(array([0.0]), array([[1.0]]))),
            MockGaussianFilter(GaussianDistribution(array([10.0]), array([[1.0]]))),
        ]
        imm = InteractingMultipleModelFilter(
            filter_bank,
            transition_matrix=array([[1.0, 0.0], [0.0, 1.0]]),
            mode_probabilities=array([0.5, 0.5]),
        )

        imm.update_linear(array([0.0]), array([[1.0]]), array([[1.0]]))

        self.assertGreater(imm.mode_probabilities[0], 0.999999)
        self.assertLess(imm.mode_probabilities[1], 1e-6)
        self.assertTrue(allclose(imm.filter_bank[0].filter_state.mu, array([0.0])))
        self.assertTrue(allclose(imm.filter_bank[1].filter_state.mu, array([5.0])))

    def test_external_mode_probability_update(self):
        filter_bank = [
            MockGaussianFilter(GaussianDistribution(array([0.0]), array([[1.0]]))),
            MockGaussianFilter(GaussianDistribution(array([1.0]), array([[1.0]]))),
        ]
        imm = InteractingMultipleModelFilter(
            filter_bank,
            transition_matrix=array([[1.0, 0.0], [0.0, 1.0]]),
            mode_probabilities=array([0.5, 0.5]),
        )

        posterior_probabilities = imm.update_mode_probabilities(
            log_likelihoods=array([0.0, -2.0])
        )

        npt.assert_allclose(
            posterior_probabilities,
            array([0.8807970779778823, 0.11920292202211755]),
        )


if __name__ == "__main__":
    unittest.main()
