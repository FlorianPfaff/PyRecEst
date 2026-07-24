# pylint: disable=no-name-in-module,no-member
import numpy as np
from pyrecest.backend import linspace, pi
from pyrecest.distributions import CircularUniformDistribution

from .abstract_sampler import AbstractSampler

_TEXT_TYPES = (str, bytes, bytearray, np.str_, np.bytes_)


def _validate_integral_scalar(value, name: str, *, minimum: int) -> int:
    scalar = np.asarray(value)
    if scalar.ndim != 0:
        raise ValueError(f"{name} must be a scalar integer")

    scalar_value = scalar.item()
    if isinstance(scalar_value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer, not a boolean")
    if isinstance(scalar_value, _TEXT_TYPES):
        raise ValueError(f"{name} must be an integer")

    try:
        integer_value = int(scalar_value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc

    try:
        is_exact_integer = bool(scalar_value == integer_value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc

    if not is_exact_integer:
        raise ValueError(f"{name} must be a finite integer")
    if integer_value < minimum:
        if minimum == 0:
            raise ValueError(f"{name} must be nonnegative")
        raise ValueError(f"{name} must be positive")
    return integer_value


class AbstractHypertoroidalSampler(AbstractSampler):
    pass


class AbstractCircularSampler(AbstractHypertoroidalSampler):
    pass


class CircularUniformSampler(AbstractCircularSampler):
    def sample_stochastic(self, n_samples: int, dim: int = 1):
        n_samples = _validate_integral_scalar(n_samples, "n_samples", minimum=0)
        dim = _validate_integral_scalar(dim, "dim", minimum=1)
        if dim != 1:
            raise ValueError(
                "CircularUniformSampler is supposed to be used for the circle "
                "(which is one-dimensional) only."
            )
        return CircularUniformDistribution().sample(n_samples)

    def get_grid(self, grid_density_parameter: int):
        """
        Returns an equidistant grid of points on the circle [0,2*pi).
        """
        grid_density_parameter = _validate_integral_scalar(
            grid_density_parameter, "grid_density_parameter", minimum=1
        )
        points = linspace(0.0, 2.0 * pi, grid_density_parameter, endpoint=False)
        # Set it to the middle of the interval instead of the start
        points += (2.0 * pi / grid_density_parameter) / 2.0
        return points
