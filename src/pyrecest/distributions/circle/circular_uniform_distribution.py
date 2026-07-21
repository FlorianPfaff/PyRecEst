import math

import numpy as np
from pyrecest.backend import array, mod, pi, to_numpy

from ..hypertorus._input_validation import as_shift_vector
from ..hypertorus.hypertoroidal_uniform_distribution import (
    HypertoroidalUniformDistribution,
)
from .abstract_circular_distribution import AbstractCircularDistribution


def _as_finite_real_scalar(value, name: str) -> float:
    """Normalize a finite real scalar accepted by every backend."""
    message = f"{name} must be a finite real scalar"
    try:
        value_array = np.asarray(to_numpy(array(value)))
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError(message) from exc

    if (
        value_array.shape != ()
        or np.issubdtype(value_array.dtype, np.bool_)
        or not np.issubdtype(value_array.dtype, np.number)
        or np.iscomplexobj(value_array)
    ):
        raise ValueError(message)

    scalar = float(value_array.item())
    if not math.isfinite(scalar):
        raise ValueError(message)
    return scalar


def _as_finite_real_array(value, name: str):
    """Normalize finite real evaluation points accepted by every backend."""
    message = f"{name} must contain only finite real values"
    try:
        backend_value = array(value)
        value_array = np.asarray(to_numpy(backend_value))
    except (TypeError, ValueError, RuntimeError, OverflowError) as exc:
        raise ValueError(message) from exc

    if (
        np.issubdtype(value_array.dtype, np.bool_)
        or not np.issubdtype(value_array.dtype, np.number)
        or np.iscomplexobj(value_array)
        or not np.all(np.isfinite(value_array))
    ):
        raise ValueError(message)
    return backend_value


class CircularUniformDistribution(
    HypertoroidalUniformDistribution, AbstractCircularDistribution
):
    """
    Circular uniform distribution
    """

    def __init__(self):
        HypertoroidalUniformDistribution.__init__(self, 1)
        AbstractCircularDistribution.__init__(self)

    def get_manifold_size(self):
        return AbstractCircularDistribution.get_manifold_size(self)

    def shift(self, shift_by):
        as_shift_vector(shift_by, self.dim)
        return CircularUniformDistribution()

    def cdf(self, xa, starting_point=0):
        """
        Evaluate cumulative distribution function

        Parameters
        ----------
        xa : (1, n)
            points where the cdf should be evaluated
        starting_point : scalar
            point where the cdf is zero (starting point can be
            [0, 2pi) on the circle, default 0

        Returns
        -------
        val : (1, n)
            cdf evaluated at columns of xa
        """

        starting_point = _as_finite_real_scalar(starting_point, "starting_point")
        xa = _as_finite_real_array(xa, "xa")
        return mod(xa - starting_point, 2.0 * pi) / (2.0 * pi)
