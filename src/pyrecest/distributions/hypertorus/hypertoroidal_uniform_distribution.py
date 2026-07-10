from typing import Union

# pylint: disable=no-name-in-module,no-member
import numpy as np
from pyrecest.backend import all as backend_all
from pyrecest.backend import (
    asarray,
    int32,
    int64,
    log,
    ndim,
    ones,
    pi,
    prod,
    random,
    zeros,
)
from pyrecest.exceptions import ShapeError

from ..abstract_uniform_distribution import AbstractUniformDistribution
from .abstract_hypertoroidal_distribution import AbstractHypertoroidalDistribution


def _validate_positive_sample_count(n) -> int:
    count_array = np.asarray(n)
    if count_array.ndim != 0:
        raise ValueError("n must be a scalar integer")

    count = count_array.item()
    if isinstance(count, (bool, np.bool_)):
        raise ValueError("n must be an integer, not a boolean")
    if isinstance(count, (str, bytes)):
        raise ValueError("n must be an integer, not text")

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


def _validate_trigonometric_moment_order(n: Union[int, int32, int64]) -> int:
    order_array = np.asarray(n)
    if order_array.ndim != 0:
        raise ValueError("n must be a scalar integer")

    order = order_array.item()
    if isinstance(order, (bool, np.bool_)):
        raise ValueError("n must be an integer, not a boolean")
    if isinstance(order, (str, bytes)):
        raise ValueError("n must be an integer, not text")

    try:
        order_int = int(order)
        order_float = float(order)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("n must be an integer") from exc

    if not np.isfinite(order_float) or not order_float.is_integer():
        raise ValueError("n must be a finite integer")
    return order_int


def _validate_pdf_inputs(xs, dim: int):
    xs = asarray(xs)
    if xs.ndim == 0:
        if dim != 1:
            raise ShapeError(
                "xs",
                xs.shape,
                expected=f"({dim},) or (n, {dim})",
                reason="scalar inputs are only valid for one-dimensional distributions",
            )
        return xs, 1, True

    if xs.ndim == 1:
        if dim == 1:
            return xs, xs.shape[0], False
        if xs.shape[0] != dim:
            raise ShapeError("xs", xs.shape, expected=f"({dim},) or (n, {dim})")
        return xs, 1, True

    if xs.shape[-1] != dim:
        raise ShapeError("xs", xs.shape, expected=f"last axis of length {dim}")
    return xs, xs.shape[0], False


def _validate_vector(name: str, value, dim: int):
    value = asarray(value)
    if value.shape != (dim,):
        raise ShapeError(name, value.shape, expected=f"({dim},)")
    return value


def _validate_boundary(name: str, value, dim: int):
    value = asarray(value)
    if ndim(value) == 0:
        if dim != 1:
            raise ShapeError(
                name,
                value.shape,
                expected=f"({dim},)",
                reason="scalar boundaries are only valid for one-dimensional distributions",
            )
        return value
    if value.shape != (dim,):
        raise ShapeError(name, value.shape, expected=f"({dim},)")
    return value


def _validate_boundary_order(left, right) -> None:
    if not bool(backend_all(right >= left)):
        raise ValueError("integration boundaries must be increasing in every dimension")


class HypertoroidalUniformDistribution(
    AbstractUniformDistribution, AbstractHypertoroidalDistribution
):
    def pdf(self, xs):
        """
        Returns the Probability Density Function evaluated at xs

        :param xs: Values at which to evaluate the PDF
        :returns: PDF evaluated at xs
        """
        _, n_inputs, single_input = _validate_pdf_inputs(xs, self.dim)

        pdf_values = 1.0 / self.get_manifold_size() * ones(n_inputs)
        return pdf_values[0] if single_input else pdf_values

    def trigonometric_moment(self, n: Union[int, int32, int64]):
        """
        Returns the n-th trigonometric moment

        :param n: Moment order
        :returns: n-th trigonometric moment
        """
        n = _validate_trigonometric_moment_order(n)
        if n == 0:
            return ones(self.dim) + 0j

        return zeros(self.dim) + 0j

    def entropy(self) -> float:
        """
        Returns the entropy of the distribution

        :returns: Entropy
        """
        return self.dim * log(2.0 * pi)

    def mean_direction(self):
        """
        Returns the mean of the circular uniform distribution.
        Since it doesn't have a unique mean, this function always raises a ValueError.

        :raises ValueError: Circular uniform distribution does not have a unique mean
        """
        raise ValueError(
            "Hypertoroidal uniform distributions do not have a unique mean"
        )

    def sample(self, n: Union[int, int32, int64]):
        """
        Returns a sample of size n from the distribution

        :param n: Sample size
        :returns: Sample of size n
        """
        n = _validate_positive_sample_count(n)
        return 2.0 * pi * random.uniform(size=(n, self.dim))

    def get_manifold_size(self):
        return (2.0 * pi) ** self.dim

    def shift(self, shift_by) -> "HypertoroidalUniformDistribution":
        """
        Shifts the distribution by shift_by.
        Since this is a uniform distribution, the shift does not change the distribution.

        :param shift_by: Angles to shift by
        :returns: Shifted distribution
        """
        _validate_vector("shift_by", shift_by, self.dim)
        return self

    def integrate(self, integration_boundaries=None) -> float:
        """
        Returns the integral of the distribution over the specified boundaries

        :param integration_boundaries: Optional boundaries for integration.
            If None, uses the entire distribution support.
        :returns: Integral over the specified boundaries
        """
        if integration_boundaries is None:
            left = zeros((self.dim,))
            right = 2.0 * pi * ones((self.dim,))
        else:
            left, right = integration_boundaries
        left = _validate_boundary("left", left, self.dim)
        right = _validate_boundary("right", right, self.dim)
        if self.dim > 1:
            _validate_boundary_order(left, right)

        volume = prod(right - left)
        return 1.0 / (2.0 * pi) ** self.dim * volume
