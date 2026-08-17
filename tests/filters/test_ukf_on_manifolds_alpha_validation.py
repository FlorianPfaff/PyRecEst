"""Regression tests for UKF-M sigma-point spread validation."""

import warnings

import numpy as np
import pytest
from pyrecest.backend import eye, zeros
from pyrecest.filters.ukf_on_manifolds import UKFOnManifolds


def _make_filter(alpha):
    def transition(state, omega, noise, dt):  # pylint: disable=unused-argument
        return state + noise

    def observation(state):
        return state

    return UKFOnManifolds(
        f=transition,
        h=observation,
        phi=lambda state, xi: state + xi,
        phi_inv=lambda reference, state: state - reference,
        Q=eye(1),
        R=eye(1),
        alpha=alpha,
        state0=zeros(1),
        P0=eye(1),
    )


@pytest.mark.parametrize(
    "alpha",
    [
        0.0,
        -1e-3,
        np.nan,
        np.inf,
        -np.inf,
        True,
        [1e-3, 0.0, 1e-3],
        [1e-3, np.nan, 1e-3],
        [1e-3, True, 1e-3],
        [1e-3, 1e-3],
        [[1e-3], [1e-3], [1e-3]],
    ],
)
def test_invalid_alpha_values_raise_parameter_error(alpha):
    with pytest.raises(ValueError, match="alpha must be a finite positive real"):
        _make_filter(alpha)


def test_masked_alpha_is_rejected_without_conversion_warning():
    masked_scalar = np.ma.array(1e-3, mask=True)
    masked_vector = np.ma.array([1e-3, 1e-3, 1e-3], mask=[False, True, False])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for alpha in (masked_scalar, masked_vector):
            with pytest.raises(
                ValueError, match="alpha must be a finite positive real"
            ):
                _make_filter(alpha)


def test_positive_scalar_and_length_three_alpha_remain_supported():
    _make_filter(np.float32(1e-3))
    _make_filter(np.array([1e-3, 2e-3, 3e-3]))
    _make_filter(np.ma.array(1e-3, mask=False))
