import copy
from collections.abc import Callable
from typing import Union

import numpy as np

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import int32, int64, zeros

from ..distributions.nonperiodic.abstract_linear_distribution import (
    AbstractLinearDistribution,
)
from ..distributions.nonperiodic.linear_dirac_distribution import (
    LinearDiracDistribution,
)
from .abstract_particle_filter import AbstractParticleFilter
from .manifold_mixins import EuclideanFilterMixin


class EuclideanParticleFilter(AbstractParticleFilter, EuclideanFilterMixin):
    """Euclidean Particle Filter Class."""

    def __init__(
        self,
        n_particles: Union[int, int32, int64],
        dim: Union[int, int32, int64],
    ):
        n_particles = self._validate_positive_int(n_particles, "n_particles")
        dim = self._validate_positive_int(dim, "dim")

        initial_distribution = LinearDiracDistribution(zeros((n_particles, dim)))
        EuclideanFilterMixin.__init__(self)
        AbstractParticleFilter.__init__(self, initial_distribution)

    @property
    def filter_state(self):
        """Get the filter state."""
        return self._filter_state

    @filter_state.setter
    def filter_state(
        self, new_state: AbstractLinearDistribution | LinearDiracDistribution
    ):
        """Set the filter state."""
        if not isinstance(new_state, LinearDiracDistribution):
            dist_dirac = LinearDiracDistribution.from_distribution(
                new_state, self._filter_state.d.shape[0]
            )
        else:
            dist_dirac = copy.deepcopy(new_state)

        if self._filter_state.d.shape != dist_dirac.d.shape:
            raise ValueError(
                "The shape of new state does not match with the existing state."
            )

        self._filter_state = dist_dirac

    def predict_nonlinear(
        self,
        f: Callable,
        noise_distribution: AbstractLinearDistribution | None = None,
        function_is_vectorized: bool = True,
        shift_instead_of_add: bool = False,
    ):
        """Predict for nonlinear system model."""
        AbstractParticleFilter.predict_nonlinear(
            self, f, noise_distribution, function_is_vectorized, shift_instead_of_add
        )

    @staticmethod
    def _validate_positive_int(value, name: str):
        message = f"{name} must be a positive integer"
        value_array = np.asarray(value)
        if (
            value_array.shape != ()
            or value_array.dtype == np.bool_
            or value_array.dtype.kind in {"M", "m"}
        ):
            raise ValueError(message)

        scalar = value_array.item()
        if isinstance(scalar, (bool, np.bool_, np.datetime64, np.timedelta64)):
            raise ValueError(message)
        if isinstance(scalar, (str, bytes, bytearray, np.str_, np.bytes_)):
            raise ValueError(message)
        if isinstance(scalar, (complex, np.complexfloating)):
            raise ValueError(message)

        try:
            integer = int(scalar)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(message) from exc
        try:
            is_exact_integer = bool(scalar == integer)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(message) from exc
        if not is_exact_integer:
            raise ValueError(message)

        if integer <= 0:
            raise ValueError(message)
        return integer
