from collections.abc import Callable
from typing import Union

import numpy as np

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    arange,
    int32,
    int64,
    linspace,
    mod,
    pi,
    tile,
)
from pyrecest.distributions import (
    AbstractHypertoroidalDistribution,
    HypertoroidalDiracDistribution,
)

from .abstract_particle_filter import AbstractParticleFilter
from .manifold_mixins import HypertoroidalFilterMixin


def _validate_positive_integer(value, name: str) -> int:
    message = f"{name} must be a positive integer."
    value_array = np.asarray(value)
    if value_array.shape != () or value_array.dtype == np.bool_:
        raise ValueError(message)

    scalar = value_array.item()
    if isinstance(scalar, (bool, np.bool_)):
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


class HypertoroidalParticleFilter(AbstractParticleFilter, HypertoroidalFilterMixin):
    def __init__(
        self,
        n_particles: Union[int, int32, int64],
        dim: Union[int, int32, int64],
    ):
        n_particles = _validate_positive_integer(n_particles, "n_particles")
        dim = _validate_positive_integer(dim, "dim")
        if dim == 1:
            points = linspace(0.0, 2.0 * pi, num=n_particles, endpoint=False)
        else:
            points = tile(
                arange(0.0, 2.0 * pi, 2.0 * pi / n_particles), (dim, 1)
            ).T.squeeze()
        filter_state = HypertoroidalDiracDistribution(points, dim=dim)
        HypertoroidalFilterMixin.__init__(self)
        AbstractParticleFilter.__init__(self, filter_state)

    def predict_nonlinear(
        self,
        f: Callable,
        noise_distribution: AbstractHypertoroidalDistribution | None = None,
        function_is_vectorized: bool = True,
        shift_instead_of_add: bool = True,
    ):
        super().predict_nonlinear(
            f,
            noise_distribution,
            function_is_vectorized,
            shift_instead_of_add,
        )
        self.filter_state.d = mod(self.filter_state.d, 2.0 * pi)

    def predict_nonlinear_nonadditive(self, f: Callable, samples, weights):
        super().predict_nonlinear_nonadditive(f, samples, weights)
        self.filter_state.d = mod(self.filter_state.d, 2.0 * pi)
