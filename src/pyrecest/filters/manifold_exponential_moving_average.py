"""Exponential moving average for states on manifolds."""

import copy
from typing import Any, Callable

import numpy as np

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import asarray

from .abstract_filter import AbstractFilter


class ManifoldExponentialMovingAverage(AbstractFilter):
    """Exponential moving average on a manifold.

    The estimate is updated by moving from the current state toward each new
    sample in the tangent space at the current estimate:

    ``x_new = phi(x, alpha * phi_inv(x, sample))``.

    Parameters
    ----------
    initial_state:
        Initial manifold element. If ``None``, the first update initializes the
        estimate directly from the first sample.
    alpha:
        Weight of the new sample. Must be in ``[0, 1]``.
    phi:
        Retraction with signature ``phi(state, tangent_vector) -> state``.
    phi_inv:
        Inverse retraction with signature
        ``phi_inv(state_ref, state) -> tangent_vector``.
    """

    def __init__(
        self,
        initial_state: Any,
        alpha: float,
        phi: Callable,
        phi_inv: Callable,
    ):
        if not callable(phi):
            raise TypeError("phi must be callable")
        if not callable(phi_inv):
            raise TypeError("phi_inv must be callable")

        self.phi = phi
        self.phi_inv = phi_inv
        self._alpha = self._validate_alpha(alpha)

        AbstractFilter.__init__(self, initial_state)

    @staticmethod
    def _validate_alpha(alpha: float) -> float:
        message = "alpha must be a real scalar between 0 and 1"
        if np.ma.is_masked(alpha):
            raise TypeError(message)
        try:
            alpha_array = np.asarray(alpha)
        except (TypeError, ValueError) as exc:
            raise TypeError(message) from exc
        if alpha_array.shape != () or alpha_array.dtype == np.bool_:
            raise TypeError(message)
        if np.issubdtype(alpha_array.dtype, np.datetime64) or np.issubdtype(
            alpha_array.dtype, np.timedelta64
        ):
            raise TypeError(message)

        alpha_scalar = alpha_array.item()
        if isinstance(
            alpha_scalar,
            (
                bool,
                np.bool_,
                str,
                bytes,
                bytearray,
                np.str_,
                np.bytes_,
                np.datetime64,
                np.timedelta64,
            ),
        ):
            raise TypeError(message)

        try:
            alpha_float = float(alpha_scalar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError(message) from exc
        if not np.isfinite(alpha_float) or alpha_float < 0.0 or alpha_float > 1.0:
            raise ValueError("alpha must be between 0 and 1")
        return alpha_float

    @property
    def alpha(self) -> float:
        """Weight assigned to each new sample."""
        return self._alpha

    @alpha.setter
    def alpha(self, alpha: float):
        self._alpha = self._validate_alpha(alpha)

    @property
    def filter_state(self):
        """Return the current manifold estimate."""
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        self._filter_state = copy.deepcopy(new_state)

    def update(self, sample):
        """Update the moving average with a new manifold-valued sample."""
        if self._filter_state is None:
            self.filter_state = sample
            return

        tangent_update = self.alpha * asarray(self.phi_inv(self._filter_state, sample))
        self.filter_state = self.phi(self._filter_state, tangent_update)

    def get_point_estimate(self):
        """Return the current manifold estimate."""
        return self._filter_state
