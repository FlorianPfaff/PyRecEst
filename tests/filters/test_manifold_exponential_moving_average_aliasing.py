"""Regression tests for manifold EMA state ownership."""

import numpy as np
import numpy.testing as npt
from pyrecest.filters import ManifoldExponentialMovingAverage


def _phi_euclidean(state, tangent):
    return state + tangent


def _phi_inv_euclidean(state_ref, state):
    return state - state_ref


def test_first_sample_is_copied_into_filter_state():
    sample = np.array([2.0, 3.0])
    ema = ManifoldExponentialMovingAverage(
        initial_state=None,
        alpha=0.5,
        phi=_phi_euclidean,
        phi_inv=_phi_inv_euclidean,
    )

    ema.update(sample)
    sample[:] = -1.0

    npt.assert_array_equal(ema.filter_state, np.array([2.0, 3.0]))


def test_explicit_state_assignment_does_not_alias_caller_array():
    ema = ManifoldExponentialMovingAverage(
        initial_state=np.array([0.0, 0.0]),
        alpha=0.5,
        phi=_phi_euclidean,
        phi_inv=_phi_inv_euclidean,
    )
    replacement = np.array([4.0, 5.0])

    ema.filter_state = replacement
    replacement[:] = 0.0

    npt.assert_array_equal(ema.filter_state, np.array([4.0, 5.0]))
