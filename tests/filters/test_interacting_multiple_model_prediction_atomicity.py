import copy
import unittest

import numpy.testing as npt

import pyrecest.backend
from pyrecest.backend import array, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.interacting_multiple_model_filter import (
    InteractingMultipleModelFilter,
)


def _double_state(state):
    return 2.0 * state


class _PredictingGaussianFilter:
    def __init__(self, mean, *, fail_on_predict=False):
        self.filter_state = GaussianDistribution(
            array([mean]), array([[1.0]]), check_validity=False
        )
        self.fail_on_predict = fail_on_predict

    def predict_identity(self, sys_noise_cov, sys_input=None):
        self.predict_linear(eye(1), sys_noise_cov, sys_input)

    def predict_linear(self, system_matrix, sys_noise_cov, sys_input=None):
        if self.fail_on_predict:
            raise RuntimeError("intentional prediction failure")
        mean = system_matrix @ self.filter_state.mu
        if sys_input is not None:
            mean = mean + sys_input
        covariance = (
            system_matrix @ self.filter_state.C @ system_matrix.T + sys_noise_cov
        )
        self.filter_state = GaussianDistribution(
            mean, covariance, check_validity=False
        )

    def predict_nonlinear(self, transition_function, sys_noise_cov, **kwargs):
        if self.fail_on_predict:
            raise RuntimeError("intentional prediction failure")
        mean = transition_function(self.filter_state.mu, **kwargs)
        covariance = self.filter_state.C + sys_noise_cov
        self.filter_state = GaussianDistribution(
            mean, covariance, check_validity=False
        )


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Only supported on numpy backend",
)
class InteractingMultipleModelPredictionAtomicityTest(unittest.TestCase):
    @staticmethod
    def _make_imm(*, fail_second=False):
        return InteractingMultipleModelFilter(
            [
                _PredictingGaussianFilter(0.0),
                _PredictingGaussianFilter(10.0, fail_on_predict=fail_second),
            ],
            transition_matrix=array([[0.75, 0.25], [0.25, 0.75]]),
            mode_probabilities=array([0.8, 0.2]),
        )

    @staticmethod
    def _snapshot(imm):
        return (
            copy.deepcopy(imm.mode_probabilities),
            [
                copy.deepcopy(curr_filter.filter_state.mu)
                for curr_filter in imm.filter_bank
            ],
            [
                copy.deepcopy(curr_filter.filter_state.C)
                for curr_filter in imm.filter_bank
            ],
            copy.deepcopy(imm.latest_mixing_probabilities),
        )

    def _assert_snapshot_equal(self, imm, snapshot):
        mode_probabilities, means, covariances, mixing = snapshot
        npt.assert_allclose(imm.mode_probabilities, mode_probabilities)
        for curr_filter, mean, covariance in zip(
            imm.filter_bank, means, covariances
        ):
            npt.assert_allclose(curr_filter.filter_state.mu, mean)
            npt.assert_allclose(curr_filter.filter_state.C, covariance)
        if mixing is None:
            self.assertIsNone(imm.latest_mixing_probabilities)
        else:
            npt.assert_allclose(imm.latest_mixing_probabilities, mixing)

    def test_invalid_prediction_arguments_do_not_apply_interaction(self):
        invalid_calls = {
            "identity": lambda imm: imm.predict_identity([array([[0.0]])]),
            "linear": lambda imm: imm.predict_linear(
                [array([[1.0]])],
                [array([[0.0]]), array([[0.0]])],
            ),
            "nonlinear": lambda imm: imm.predict_nonlinear(
                [_double_state],
                [array([[0.0]]), array([[0.0]])],
            ),
        }

        for prediction_name, invalid_call in invalid_calls.items():
            with self.subTest(prediction=prediction_name):
                imm = self._make_imm()
                imm.latest_mixing_probabilities = array(
                    [[0.6, 0.4], [0.3, 0.7]]
                )
                before = self._snapshot(imm)

                with self.assertRaisesRegex(ValueError, "one entry per model"):
                    invalid_call(imm)

                self._assert_snapshot_equal(imm, before)

    def test_subfilter_failure_rolls_back_interaction_and_predictions(self):
        failing_calls = {
            "identity": lambda imm: imm.predict_identity(
                [array([[0.5]]), array([[0.25]])]
            ),
            "linear": lambda imm: imm.predict_linear(
                [array([[2.0]]), array([[3.0]])],
                [array([[0.5]]), array([[0.25]])],
            ),
            "nonlinear": lambda imm: imm.predict_nonlinear(
                [_double_state, _double_state],
                [array([[0.5]]), array([[0.25]])],
            ),
        }

        for prediction_name, failing_call in failing_calls.items():
            with self.subTest(prediction=prediction_name):
                imm = self._make_imm(fail_second=True)
                imm.latest_mixing_probabilities = array(
                    [[0.55, 0.45], [0.35, 0.65]]
                )
                before = self._snapshot(imm)

                with self.assertRaisesRegex(
                    RuntimeError, "intentional prediction failure"
                ):
                    failing_call(imm)

                self._assert_snapshot_equal(imm, before)


if __name__ == "__main__":
    unittest.main()
