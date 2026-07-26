import copy
import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member,unused-argument
import pyrecest.backend
from pyrecest.backend import array, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.interacting_multiple_model_filter import (
    InteractingMultipleModelFilter,
)


class MutatingUpdateFilter:
    def __init__(
        self,
        initial_state: GaussianDistribution,
        *,
        fail_linear: bool = False,
        fail_nonlinear: bool = False,
    ):
        self.filter_state = copy.deepcopy(initial_state)
        self.fail_linear = fail_linear
        self.fail_nonlinear = fail_nonlinear

    def update_linear(self, measurement, measurement_matrix, meas_noise):
        self.filter_state = GaussianDistribution(
            self.filter_state.mu + 1.0,
            self.filter_state.C + 1.0,
            check_validity=False,
        )
        if self.fail_linear:
            raise RuntimeError("linear update failed")

    def update_nonlinear(
        self,
        measurement,
        measurement_function,
        meas_noise,
        **kwargs,
    ):
        self.filter_state = GaussianDistribution(
            self.filter_state.mu + 1.0,
            self.filter_state.C + 1.0,
            check_validity=False,
        )
        if self.fail_nonlinear:
            raise RuntimeError("nonlinear update failed")


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Only supported on numpy backend",
)
class ImmUpdateTransactionTest(unittest.TestCase):
    @staticmethod
    def _make_imm(*, fail_linear=False, fail_nonlinear=False):
        filter_bank = [
            MutatingUpdateFilter(
                GaussianDistribution(array([0.0]), array([[1.0]])),
            ),
            MutatingUpdateFilter(
                GaussianDistribution(array([2.0]), array([[1.0]])),
                fail_linear=fail_linear,
                fail_nonlinear=fail_nonlinear,
            ),
        ]
        return InteractingMultipleModelFilter(
            filter_bank,
            transition_matrix=eye(2),
            mode_probabilities=array([0.4, 0.6]),
        )

    def _assert_update_rolled_back(
        self,
        imm,
        previous_states,
        previous_probabilities,
        previous_likelihoods,
        previous_log_likelihoods,
    ):
        for curr_filter, previous_state in zip(imm.filter_bank, previous_states):
            npt.assert_allclose(curr_filter.filter_state.mu, previous_state.mu)
            npt.assert_allclose(curr_filter.filter_state.C, previous_state.C)
        npt.assert_allclose(imm.mode_probabilities, previous_probabilities)
        npt.assert_allclose(imm.latest_model_likelihoods, previous_likelihoods)
        npt.assert_allclose(
            imm.latest_log_model_likelihoods,
            previous_log_likelihoods,
        )

    def test_linear_update_rolls_back_all_models_on_failure(self):
        imm = self._make_imm(fail_linear=True)
        imm.latest_model_likelihoods = array([0.25, 0.75])
        imm.latest_log_model_likelihoods = array([-1.0, -0.5])
        previous_states = [
            copy.deepcopy(curr_filter.filter_state) for curr_filter in imm.filter_bank
        ]
        previous_probabilities = copy.deepcopy(imm.mode_probabilities)
        previous_likelihoods = copy.deepcopy(imm.latest_model_likelihoods)
        previous_log_likelihoods = copy.deepcopy(imm.latest_log_model_likelihoods)

        with self.assertRaisesRegex(RuntimeError, "linear update failed"):
            imm.update_linear(
                array([0.0]),
                eye(1),
                eye(1),
            )

        self._assert_update_rolled_back(
            imm,
            previous_states,
            previous_probabilities,
            previous_likelihoods,
            previous_log_likelihoods,
        )

    def test_nonlinear_update_rolls_back_all_models_on_failure(self):
        imm = self._make_imm(fail_nonlinear=True)
        imm.latest_model_likelihoods = array([0.3, 0.7])
        imm.latest_log_model_likelihoods = array([-1.2, -0.4])
        previous_states = [
            copy.deepcopy(curr_filter.filter_state) for curr_filter in imm.filter_bank
        ]
        previous_probabilities = copy.deepcopy(imm.mode_probabilities)
        previous_likelihoods = copy.deepcopy(imm.latest_model_likelihoods)
        previous_log_likelihoods = copy.deepcopy(imm.latest_log_model_likelihoods)

        with self.assertRaisesRegex(RuntimeError, "nonlinear update failed"):
            imm.update_nonlinear(
                array([0.0]),
                lambda state: state,
                eye(1),
                likelihoods=array([0.8, 0.2]),
            )

        self._assert_update_rolled_back(
            imm,
            previous_states,
            previous_probabilities,
            previous_likelihoods,
            previous_log_likelihoods,
        )


if __name__ == "__main__":
    unittest.main()
