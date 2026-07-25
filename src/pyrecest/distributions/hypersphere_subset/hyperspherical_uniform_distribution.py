from typing import Union

# pylint: disable=no-name-in-module,no-member
import numpy as np
from pyrecest.backend import (
    array,
    cos,
    empty,
    int32,
    int64,
    linalg,
    ones,
    pi,
)
from pyrecest.backend import random as backend_random
from pyrecest.backend import (
    sin,
    sqrt,
    stack,
)

from .abstract_hypersphere_subset_uniform_distribution import (
    AbstractHypersphereSubsetUniformDistribution,
)
from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution


def _validate_positive_sample_count(n) -> int:
    count_array = np.asarray(n)
    if count_array.ndim != 0:
        raise ValueError("n must be a scalar integer")

    count = count_array.item()
    if isinstance(count, (bool, np.bool_)):
        raise ValueError("n must be an integer, not a boolean")
    if isinstance(count, (str, bytes)):
        raise ValueError("n must be an integer")

    try:
        count_int = int(count)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("n must be an integer") from exc

    try:
        is_exact_integer = count == count_int
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("n must be an integer") from exc
    if not bool(is_exact_integer):
        raise ValueError("n must be a finite integer")
    if count_int <= 0:
        raise ValueError("n must be positive")
    return count_int


class HypersphericalUniformDistribution(
    AbstractHypersphericalDistribution, AbstractHypersphereSubsetUniformDistribution
):
    def __init__(self, dim: Union[int, int32, int64]):
        AbstractHypersphereSubsetUniformDistribution.__init__(self, dim)

    def pdf(self, xs):
        return AbstractHypersphereSubsetUniformDistribution.pdf(self, xs)

    def ln_pdf(self, xs):
        xs = array(xs)
        if xs.ndim == 0 or xs.shape[-1] != self.input_dim:
            raise ValueError("Invalid shape of input data points.")
        log_density = -self.get_ln_manifold_size()
        return log_density * ones(xs.shape[:-1]) if xs.ndim > 1 else log_density

    def sample(self, n: Union[int, int32, int64]):
        n = _validate_positive_sample_count(n)

        if self.dim == 2:
            s = empty(
                (
                    n,
                    self.dim + 1,
                )
            )
            phi = 2.0 * pi * backend_random.uniform(size=n)
            sz = backend_random.uniform(size=n) * 2.0 - 1.0
            r = sqrt(1 - sz**2)
            s = stack([r * cos(phi), r * sin(phi), sz], axis=1)
        else:
            samples_unnorm = backend_random.normal(size=(n, self.dim + 1))
            s = samples_unnorm / linalg.norm(samples_unnorm, axis=1).reshape(-1, 1)
        return s

    def get_manifold_size(self):
        return AbstractHypersphericalDistribution.get_manifold_size(self)
