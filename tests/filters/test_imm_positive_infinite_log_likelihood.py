import numpy as np
import numpy.testing as npt
import pytest
from pyrecest.backend import copy as backend_copy
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import InteractingMultipleModelFilter


class _StaticGaussianFilter:
    def __init__(self, mean: float):
        self.filter_state = GaussianDistribution(
            np.array([mean]),
            np.array([[1.0]]),
            check_validity=False,
        )


def _make_imm() -> InteractingMultipleModelFilter:
    return InteractingMultipleModelFilter(
        [_StaticGaussianFilter(0.0), _StaticGaussianFilter(1.0)],
        transition_matrix=np.eye(2),
        mode_probabilities=np.array([0.4, 0.6]),
    )


def test_positive_infinite_log_likelihood_is_rejected_atomically():
    imm = _make_imm()
    imm.latest_model_likelihoods = np.array([0.25, 0.75])
    imm.latest_log_model_likelihoods = np.log(imm.latest_model_likelihoods)

    expected_probabilities = backend_copy(imm.mode_probabilities)
    expected_likelihoods = backend_copy(imm.latest_model_likelihoods)
    expected_log_likelihoods = backend_copy(imm.latest_log_model_likelihoods)

    with pytest.raises(ValueError, match="positive infinity"):
        imm.update_mode_probabilities(log_likelihoods=np.array([float("inf"), 0.0]))

    npt.assert_array_equal(imm.mode_probabilities, expected_probabilities)
    npt.assert_array_equal(imm.latest_model_likelihoods, expected_likelihoods)
    npt.assert_array_equal(
        imm.latest_log_model_likelihoods,
        expected_log_likelihoods,
    )


def test_negative_infinite_log_likelihood_remains_zero_mass():
    imm = _make_imm()

    posterior = imm.update_mode_probabilities(
        log_likelihoods=np.array([-float("inf"), 0.0])
    )

    npt.assert_array_equal(posterior, np.array([0.0, 1.0]))
    npt.assert_array_equal(
        imm.latest_model_likelihoods,
        np.array([0.0, 1.0]),
    )
