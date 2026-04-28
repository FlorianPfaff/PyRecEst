"""Sliding-window smoother based on manifold means."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial

from pyrecest.backend import any as backend_any
from pyrecest.backend import asarray, concatenate, ndim, stack, sum
from pyrecest.distributions import (
    AbstractHypercylindricalDistribution,
    AbstractHyperhemisphericalDistribution,
    AbstractHypersphericalDistribution,
    AbstractHypertoroidalDistribution,
    AbstractLinearDistribution,
    AbstractSE2Distribution,
    AbstractSE3Distribution,
    CircularDiracDistribution,
    HypercylindricalDiracDistribution,
    HyperhemisphericalDiracDistribution,
    HypersphericalDiracDistribution,
    HypertoroidalDiracDistribution,
    LinearDiracDistribution,
    SE2DiracDistribution,
    SE3DiracDistribution,
)

from .abstract_smoother import AbstractSmoother


class SlidingWindowManifoldMeanSmoother(AbstractSmoother):
    """Smooth state estimates by replacing each entry with a local manifold mean.

    The smoother extracts a representative point from each input state. For
    PyRecEst distributions this is ``state.mean()``; raw arrays are used directly.
    Each sliding window is then represented as a Dirac distribution on the same
    manifold, and the Dirac distribution's ``mean()`` defines the smoothed value.

    Parameters
    ----------
    window_size
        Number of sequence entries considered for each smoothed state. Edge
        windows are truncated to the available samples.
    dirac_distribution_factory
        Optional callable ``factory(points, weights)`` used to construct the
        window Dirac distribution. This is useful for raw samples on non-linear
        manifolds. If omitted, the smoother infers a factory from distribution
        inputs and falls back to a Euclidean Dirac distribution for raw arrays.
    window_weights
        Optional non-negative weights with length ``window_size``. Truncated edge
        windows use the corresponding weight slice and are renormalized by the
        Dirac distribution.
    alignment
        ``"center"`` uses past and future estimates around each state,
        ``"trailing"`` uses the current and previous estimates, and ``"leading"``
        uses the current and following estimates.
    """

    _ALIGNMENTS = ("center", "trailing", "leading")

    def __init__(
        self,
        window_size: int = 3,
        dirac_distribution_factory: Callable | None = None,
        window_weights=None,
        alignment: str = "center",
    ):
        window_size = int(window_size)
        if window_size < 1:
            raise ValueError("window_size must be a positive integer.")
        if alignment not in self._ALIGNMENTS:
            raise ValueError(f"alignment must be one of {', '.join(self._ALIGNMENTS)}.")

        self.window_size = window_size
        self.dirac_distribution_factory = dirac_distribution_factory
        self.alignment = alignment

        if window_weights is None:
            self.window_weights = None
        else:
            self.window_weights = asarray(window_weights).reshape(-1)
            if self.window_weights.shape[0] != self.window_size:
                raise ValueError("window_weights must have length window_size.")
            if backend_any(self.window_weights < 0):
                raise ValueError("window_weights must be non-negative.")
            if sum(self.window_weights) <= 0:
                raise ValueError("window_weights must contain a positive weight.")

    @staticmethod
    def _as_vector(value):
        if isinstance(value, tuple):
            return concatenate([asarray(part).reshape(-1) for part in value])

        value_array = asarray(value)
        if ndim(value_array) == 0:
            return value_array.reshape((1,))
        return value_array.reshape(-1)

    def _state_to_value(self, state):
        mean = getattr(state, "mean", None)
        if callable(mean):
            return self._as_vector(mean())
        return self._as_vector(state)

    def _window_bounds(self, time_idx: int, sequence_length: int):
        if self.alignment == "center":
            past = self.window_size // 2
            future = self.window_size - past - 1
            raw_start = time_idx - past
            raw_end = time_idx + future + 1
        elif self.alignment == "trailing":
            raw_start = time_idx - self.window_size + 1
            raw_end = time_idx + 1
        else:
            raw_start = time_idx
            raw_end = time_idx + self.window_size

        start = max(raw_start, 0)
        end = min(raw_end, sequence_length)
        weight_start = start - raw_start
        weight_end = weight_start + end - start

        return start, end, weight_start, weight_end

    def _weights_for_window(self, weight_start: int, weight_end: int):
        if self.window_weights is None:
            return None

        weights = self.window_weights[weight_start:weight_end]
        weights_sum = sum(weights)
        if weights_sum <= 0:
            raise ValueError("At least one active window weight must be positive.")
        return weights / weights_sum

    @staticmethod
    def _circular_dirac_factory(points, weights):
        return CircularDiracDistribution(points.reshape(-1), weights)

    @classmethod
    def _default_distribution_factory(cls, state):
        simple_factories = (
            (AbstractSE2Distribution, SE2DiracDistribution),
            (AbstractSE3Distribution, SE3DiracDistribution),
            (
                AbstractHyperhemisphericalDistribution,
                HyperhemisphericalDiracDistribution,
            ),
            (AbstractHypersphericalDistribution, HypersphericalDiracDistribution),
        )
        for distribution_type, factory in simple_factories:
            if isinstance(state, distribution_type):
                return factory

        if isinstance(state, AbstractHypercylindricalDistribution):
            return partial(HypercylindricalDiracDistribution, state.bound_dim)

        if isinstance(state, AbstractHypertoroidalDistribution):
            if state.dim == 1:
                return cls._circular_dirac_factory
            return partial(HypertoroidalDiracDistribution, dim=state.dim)

        if isinstance(state, AbstractLinearDistribution):
            return LinearDiracDistribution

        return LinearDiracDistribution

    def smooth(self, states: Sequence) -> list:
        """Return smoothed manifold mean values for a sequence of states."""

        states_list = list(states)
        if len(states_list) == 0:
            return []

        state_values = [self._state_to_value(state) for state in states_list]
        distribution_factory = (
            self.dirac_distribution_factory
            if self.dirac_distribution_factory is not None
            else self._default_distribution_factory(states_list[0])
        )

        smoothed_values = []
        for time_idx in range(len(state_values)):
            start, end, weight_start, weight_end = self._window_bounds(
                time_idx, len(state_values)
            )
            window_values = stack(state_values[start:end], axis=0)
            window_weights = self._weights_for_window(weight_start, weight_end)
            window_distribution = distribution_factory(window_values, window_weights)
            smoothed_values.append(self._as_vector(window_distribution.mean()))

        return smoothed_values
