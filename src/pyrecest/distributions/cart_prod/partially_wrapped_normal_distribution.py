import copy
from numbers import Integral
from typing import Union

import numpy as np

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import all as backend_all
from pyrecest.backend import (
    allclose,
    arange,
    array,
    concatenate,
    cos,
    diag,
    exp,
    hstack,
    int32,
    int64,
    isfinite,
    linalg,
    meshgrid,
    mod,
    pi,
    random,
    repeat,
    reshape,
    sin,
    sum,
    tile,
    where,
)

from ..hypertorus.hypertoroidal_wrapped_normal_distribution import (
    HypertoroidalWrappedNormalDistribution,
)
from ..nonperiodic.gaussian_distribution import GaussianDistribution
from .abstract_hypercylindrical_distribution import AbstractHypercylindricalDistribution


def _validate_positive_sample_count(n) -> int:
    count_array = np.asarray(n)
    if count_array.ndim != 0:
        raise ValueError("n must be a scalar integer")
    if count_array.dtype.kind in {"c", "M", "m"}:
        raise ValueError("n must be a finite integer")

    count = count_array.item()
    if isinstance(count, (bool, np.bool_)):
        raise ValueError("n must be an integer, not a boolean")
    if isinstance(
        count,
        (complex, np.complexfloating, np.datetime64, np.timedelta64),
    ):
        raise ValueError("n must be a finite integer")

    try:
        count_int = int(count)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("n must be an integer") from exc

    try:
        is_exact_integer = count == count_int
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("n must be a finite integer") from exc
    if not bool(is_exact_integer):
        raise ValueError("n must be a finite integer")
    if count_int <= 0:
        raise ValueError("n must be positive")
    return count_int


def _validate_nonnegative_wrap_count(m) -> int:
    count_array = np.asarray(m)
    if count_array.ndim != 0:
        raise ValueError("m must be a scalar integer")
    if count_array.dtype.kind in {"c", "M", "m"}:
        raise ValueError("m must be a finite integer")

    count = count_array.item()
    if isinstance(count, (bool, np.bool_)):
        raise ValueError("m must be an integer, not a boolean")
    if isinstance(
        count,
        (complex, np.complexfloating, np.datetime64, np.timedelta64),
    ):
        raise ValueError("m must be a finite integer")

    try:
        count_int = int(count)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("m must be an integer") from exc

    try:
        is_exact_integer = count == count_int
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError("m must be a finite integer") from exc
    if not bool(is_exact_integer):
        raise ValueError("m must be a finite integer")
    if count_int < 0:
        raise ValueError("m must be non-negative")
    return count_int


def _as_2d_query_points(xs, dim: int):
    """Normalize query points and retain their density output shape."""
    xs = array(xs)
    if xs.ndim == 0:
        if dim != 1:
            raise ValueError(
                "Scalar query points are only valid for one-dimensional "
                f"distributions, got dim={dim}."
            )
        return reshape(xs, (1, 1)), (1,)

    if xs.ndim == 1:
        if dim == 1:
            return reshape(xs, (-1, 1)), xs.shape
        if xs.shape[0] == dim:
            return reshape(xs, (1, dim)), (1,)
        raise ValueError(
            "Last dimension of xs must match the distribution dimension "
            f"{dim}, got shape {xs.shape}."
        )

    if xs.shape[-1] != dim:
        raise ValueError(
            "Last dimension of xs must match the distribution dimension "
            f"{dim}, got shape {xs.shape}."
        )
    return reshape(xs, (-1, dim)), xs.shape[:-1]


def _validate_bound_dim(bound_dim, total_dim: int) -> int:
    if isinstance(bound_dim, bool) or not isinstance(bound_dim, Integral):
        raise ValueError("bound_dim must be a non-negative integer")
    bound_dim = int(bound_dim)
    if bound_dim < 0:
        raise ValueError("bound_dim must be non-negative")
    if bound_dim > total_dim:
        raise ValueError("bound_dim must not exceed the distribution dimension")
    return bound_dim


class PartiallyWrappedNormalDistribution(AbstractHypercylindricalDistribution):
    """Partially wrapped normal distribution on periodic-linear domains.

    References
    ----------
    Kurz, G., Gilitschenski, I., & Hanebeck, U. D. (2014). The Partially
    Wrapped Normal Distribution for SE(2) Estimation. Proceedings of the
    2014 IEEE International Conference on Multisensor Fusion and Information
    Integration.
    """

    def __init__(self, mu, C, bound_dim: Union[int, int32, int64]):
        mu = array(mu)
        C = array(C)
        if mu.ndim != 1:
            raise ValueError("mu must be a 1-dimensional array")
        bound_dim = _validate_bound_dim(bound_dim, mu.shape[0])
        if C.shape != (mu.shape[-1], mu.shape[-1]):
            raise ValueError("C must match size of mu")
        if not bool(allclose(C, C.T)):
            raise ValueError("C must be symmetric")
        try:
            chol = linalg.cholesky(C)
        except Exception as exc:
            raise ValueError("C must be positive definite") from exc
        if not bool(backend_all(isfinite(chol))):
            raise ValueError("C must be positive definite")
        mu = where(arange(mu.shape[0]) < bound_dim, mod(mu, 2 * pi), mu)

        AbstractHypercylindricalDistribution.__init__(
            self, bound_dim=bound_dim, lin_dim=mu.shape[0] - bound_dim
        )

        self.mu = where(arange(mu.shape[0]) < bound_dim, mod(mu, 2.0 * pi), mu)
        self.C = C

    def pdf(self, xs, m: Union[int, int32, int64] = 3):
        m = _validate_nonnegative_wrap_count(m)
        xs, output_shape = _as_2d_query_points(xs, self.input_dim)
        condition = (
            arange(xs.shape[1]) < self.bound_dim
        )  # Create a condition based on column indices
        xs = where(
            # Broadcast the condition to match the shape of xs
            condition[None, :],  # noqa: E203
            mod(xs, 2.0 * pi),  # Compute the modulus where the condition is True
            xs,  # Keep the original values where the condition is False
        )

        # generate multiples for wrapping
        multiples = array(range(-m, m + 1)) * 2.0 * pi

        # create meshgrid for all combinations of multiples
        if self.bound_dim == 0:
            mesh = array([[]])
        else:
            mesh = array(
                meshgrid(*[multiples] * self.bound_dim, indexing="ij")
            ).reshape(-1, self.bound_dim)

        # reshape xs for broadcasting: repeat each row mesh.shape[0] times so that
        # every xs[i] is paired with every mesh offset before moving to xs[i+1]
        xs_reshaped = repeat(
            xs[:, : self.bound_dim], mesh.shape[0], axis=0
        )  # noqa: E203

        # prepare data for wrapping (not applied to linear dimensions)
        xs_wrapped = xs_reshaped + tile(mesh, (xs.shape[0], 1))
        xs_wrapped = concatenate(
            [
                xs_wrapped,
                repeat(xs[:, self.bound_dim :], mesh.shape[0], axis=0),  # noqa: E203
            ],
            axis=1,
        )

        # Evaluate the Gaussian factor without leaving the active backend.
        evals = GaussianDistribution(self.mu, self.C, check_validity=False).pdf(
            xs_wrapped
        )

        # sum evaluations for the wrapped dimensions
        summed_evals = sum(evals.reshape(-1, (2 * m + 1) ** self.bound_dim), axis=1)

        return reshape(summed_evals, output_shape)

    def mode(self):
        """
        Determines the mode of the distribution, i.e., the point where the pdf is largest.
        Returns:
            m (lin_dim + bound_dim,) vector: the mode
        """
        return self.mu

    def set_mean(self, new_mean):
        """
        Return a copy of this distribution with the location parameter shifted to ``new_mean``.

        For bounded dimensions, the mean is wrapped into [0, 2*pi) to stay on the manifold.
        """
        new_mean = array(new_mean)
        if new_mean.shape != (self.input_dim,):
            raise ValueError("new_mean must match distribution dim")
        new_dist = copy.deepcopy(self)
        wrapped_mean = where(
            arange(new_mean.shape[0]) < self.bound_dim, mod(new_mean, 2 * pi), new_mean
        )
        new_dist.mu = wrapped_mean
        return new_dist

    def set_mode(self, new_mode):
        return self.set_mean(new_mode)

    def hybrid_moment(self):
        """Return periodic trigonometric moments followed by linear means.

        The periodic entries follow the package-wide hypercylindrical convention:
        all cosine moments, then all sine moments, followed by the linear means.
        """
        mu_lin = self.mu[self.bound_dim :]  # noqa: E203

        mu_bound_odd = sin(self.mu[: self.bound_dim]) * exp(
            -diag(self.C)[: self.bound_dim] / 2
        )
        mu_bound_even = cos(self.mu[: self.bound_dim]) * exp(
            -diag(self.C)[: self.bound_dim] / 2
        )

        mu_bound = hstack((mu_bound_even, mu_bound_odd))

        return hstack((mu_bound, mu_lin))

    def hybrid_mean(self):
        return self.mu

    def linear_mean(self):
        return self.mu[self.bound_dim :]  # noqa: E203

    def periodic_mean(self):
        return self.mu[: self.bound_dim]

    def sample(self, n: int):
        """
        Sample n points from the distribution
        Parameters:
            n (int): number of points to sample
        """
        n = _validate_positive_sample_count(n)
        s = random.multivariate_normal(mean=self.mu, cov=self.C, size=(n,))
        wrapped_values = mod(s[:, : self.bound_dim], 2.0 * pi)
        unbounded_values = s[:, self.bound_dim :]  # noqa: E203

        # Concatenate the modified section with the unmodified section
        s = concatenate([wrapped_values, unbounded_values], axis=1)
        return s

    def to_gaussian(self):
        return GaussianDistribution(self.mu, self.C)

    def linear_covariance(self, approximate_mean=None):
        _ = approximate_mean
        return self.C[self.bound_dim :, self.bound_dim :]  # noqa: E203

    def marginalize_periodic(self):
        return GaussianDistribution(
            self.mu[self.bound_dim :],  # noqa: E203
            self.C[self.bound_dim :, self.bound_dim :],  # noqa: E203
        )

    def marginalize_linear(self):
        return HypertoroidalWrappedNormalDistribution(
            self.mu[: self.bound_dim],
            self.C[: self.bound_dim, : self.bound_dim],  # noqa: E203
        )
