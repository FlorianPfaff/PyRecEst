# pylint: disable=no-name-in-module,no-member
import numpy as np
from pyrecest.backend import asarray, concatenate, hstack, prod, reshape, stack

from .abstract_cart_prod_distribution import AbstractCartProdDistribution


def _validate_positive_sample_count(n) -> int:
    count_array = np.asarray(n)
    if count_array.ndim != 0:
        raise ValueError("n must be a scalar integer")

    count = count_array.item()
    if isinstance(count, (bool, np.bool_)):
        raise ValueError("n must be an integer, not a boolean")

    try:
        count_int = int(count)
        count_float = float(count)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("n must be an integer") from exc

    if not np.isfinite(count_float) or not count_float.is_integer():
        raise ValueError("n must be a finite integer")
    if count_int <= 0:
        raise ValueError("n must be positive")
    return count_int


class CartProdStackedDistribution(AbstractCartProdDistribution):
    def __init__(self, dists):
        self.dists = dists
        AbstractCartProdDistribution.__init__(self, sum(dist.dim for dist in dists))

    @property
    def input_dim(self) -> int:
        return sum(dist.input_dim for dist in self.dists)

    def get_manifold_size(self) -> float:
        size = 1.0
        for dist in self.dists:
            size *= float(dist.get_manifold_size())
        return size

    def sample(self, n: int):
        n = _validate_positive_sample_count(n)
        return hstack([dist.sample(n) for dist in self.dists])

    def pdf(self, xs):
        xs = asarray(xs)
        ps = []
        next_dim = 0
        for dist in self.dists:
            next_input_dim = next_dim + dist.input_dim
            if xs.ndim == 1:
                xs_curr = xs[next_dim:next_input_dim]
            else:
                xs_curr = xs[:, next_dim:next_input_dim]
            pdf_value = asarray(dist.pdf(xs_curr))
            if xs.ndim == 1:
                pdf_value = reshape(pdf_value, ())
            ps.append(pdf_value)
            next_dim = next_input_dim
        return prod(stack(ps), axis=0)

    def shift(self, shift_by):
        if len(shift_by) != self.dim:
            raise ValueError("Incorrect number of offsets.")
        shifted_dists = []
        curr_dim = 0
        for dist in self.dists:
            shifted_dists.append(
                dist.shift(shift_by[curr_dim : curr_dim + dist.dim])  # noqa: E203
            )
            curr_dim += dist.dim
        return self.__class__(shifted_dists)

    def set_mode(self, new_mode):
        new_dists = []
        curr_ind = 0
        for dist in self.dists:
            next_ind = curr_ind + dist.input_dim
            new_dists.append(dist.set_mode(new_mode[curr_ind:next_ind]))
            curr_ind = next_ind
        return self.__class__(new_dists)

    def hybrid_mean(self):
        return concatenate([dist.mean() for dist in self.dists])

    def mean(self):
        """
        Convenient access to hybrid_mean() to have a consistent interface
        throughout manifolds.

        :return: The mean of the distribution.
        :rtype:
        """
        return self.hybrid_mean()

    def mode(self):
        return concatenate([dist.mode() for dist in self.dists])
