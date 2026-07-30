import numpy as np
import numpy.testing as npt
import pyrecest.backend
import pytest
from pyrecest.backend import array
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.interacting_multiple_model_filter import (
    InteractingMultipleModelFilter,
)


class _GaussianStateHolder:
    def __init__(self, mean):
        self.filter_state = GaussianDistribution(
            array([mean]),
            array([[1.0]]),
            check_validity=False,
        )


def _filter_bank():
    return [_GaussianStateHolder(0.0), _GaussianStateHolder(2.0)]


@pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Extreme binary64 regression is specific to the NumPy backend",
)
def test_extreme_finite_mode_probabilities_preserve_ratios():
    largest = np.finfo(np.float64).max

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        with pytest.warns(UserWarning, match="mode_probabilities"):
            imm = InteractingMultipleModelFilter(
                _filter_bank(),
                transition_matrix=array([[1.0, 0.0], [0.0, 1.0]]),
                mode_probabilities=array([largest, largest / 2.0]),
            )

    npt.assert_allclose(imm.mode_probabilities, array([2.0 / 3.0, 1.0 / 3.0]))
    npt.assert_allclose(imm.get_point_estimate(), array([2.0 / 3.0]))


@pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Extreme binary64 regression is specific to the NumPy backend",
)
def test_extreme_finite_transition_rows_remain_stochastic():
    largest = np.finfo(np.float64).max

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        with pytest.warns(UserWarning, match="transition_matrix"):
            imm = InteractingMultipleModelFilter(
                _filter_bank(),
                transition_matrix=array(
                    [[largest, largest], [largest, 0.0]],
                ),
                mode_probabilities=array([0.25, 0.75]),
            )
        mixing = imm.interact()

    npt.assert_allclose(
        imm.transition_matrix,
        array([[0.5, 0.5], [1.0, 0.0]]),
    )
    npt.assert_allclose(imm.mode_probabilities, array([0.875, 0.125]))
    npt.assert_allclose(mixing.sum(axis=0), array([1.0, 1.0]))


@pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Extreme binary64 regression is specific to the NumPy backend",
)
def test_moment_matching_normalizes_extreme_weights_and_rejects_negative_weights():
    largest = np.finfo(np.float64).max
    gaussians = [holder.filter_state for holder in _filter_bank()]

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        matched = InteractingMultipleModelFilter._moment_match_gaussians(
            gaussians,
            array([largest, largest]),
        )

    npt.assert_allclose(matched.mu, array([1.0]))
    npt.assert_allclose(matched.C, array([[2.0]]))

    with pytest.raises(ValueError, match="nonnegative"):
        InteractingMultipleModelFilter._moment_match_gaussians(
            gaussians,
            array([-1.0, 2.0]),
        )
