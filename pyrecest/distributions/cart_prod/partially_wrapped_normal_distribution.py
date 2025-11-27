import copy
from math import pi
from typing import Union

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    allclose,
    arange,
    array,
    atleast_2d,
    concatenate,
    cos,
    diag,
    exp,
    hstack,
    int32,
    int64,
    linalg,
    meshgrid,
    mod,
    ndim,
    random,
    repeat,
    sin,
    stack,
    sum,
    tile,
    where,
)
from scipy.stats import multivariate_normal
from itertools import product

from ..hypertorus.hypertoroidal_wrapped_normal_distribution import (
    HypertoroidalWrappedNormalDistribution,
)
from ..nonperiodic.gaussian_distribution import GaussianDistribution
from .abstract_hypercylindrical_distribution import AbstractHypercylindricalDistribution


class PartiallyWrappedNormalDistribution(AbstractHypercylindricalDistribution):
    def __init__(self, mu, C, bound_dim: Union[int, int32, int64]):
        assert bound_dim >= 0, "bound_dim must be non-negative"
        assert ndim(mu) == 1, "mu must be a 1-dimensional array"
        assert C.shape == (mu.shape[-1], mu.shape[-1]), "C must match size of mu"
        assert allclose(C, C.T), "C must be symmetric"
        assert (
            len(linalg.cholesky(C)) > 0
        ), "C must be positive definite"  # Will fail if not positive definite
        assert bound_dim <= mu.shape[0]
        mu = where(arange(mu.shape[0]) < bound_dim, mod(mu, 2 * pi), mu)

        AbstractHypercylindricalDistribution.__init__(
            self, bound_dim=bound_dim, lin_dim=mu.shape[0] - bound_dim
        )

        self.mu = where(arange(mu.shape[0]) < bound_dim, mod(mu, 2.0 * pi), mu)
        self.C = C

    def pdf(self, xs, m: Union[int, int32, int64] = 3):
        xs = atleast_2d(xs)
        condition = (
            arange(xs.shape[1]) < self.bound_dim
        )  # Create a condition based on column indices
        xs = where(
            # Broadcast the condition to match the shape of xs
            condition[None, :],  # noqa: E203
            mod(xs, 2.0 * pi),  # Compute the modulus where the condition is True
            xs,  # Keep the original values where the condition is False
        )

        assert xs.shape[-1] == self.input_dim

        # generate multiples for wrapping
        multiples = array(range(-m, m + 1)) * 2.0 * pi

        # create meshgrid for all combinations of multiples
        mesh = array(meshgrid(*[multiples] * self.bound_dim)).reshape(
            -1, self.bound_dim
        )

        # reshape xs for broadcasting
        xs_reshaped = tile(xs[:, : self.bound_dim], (mesh.shape[0], 1))  # noqa: E203

        # prepare data for wrapping (not applied to linear dimensions)
        xs_wrapped = xs_reshaped + repeat(mesh, xs.shape[0], axis=0)
        xs_wrapped = concatenate(
            [
                xs_wrapped,
                tile(xs[:, self.bound_dim :], (mesh.shape[0], 1)),  # noqa: E203
            ],
            axis=1,
        )

        # evaluate normal for all xs_wrapped
        mvn = multivariate_normal(self.mu, self.C)
        evals = array(mvn.pdf(xs_wrapped))  # For being compatible with all backends

        # sum evaluations for the wrapped dimensions
        summed_evals = sum(evals.reshape(-1, (2 * m + 1) ** self.bound_dim), axis=1)

        return summed_evals

    def mode(self):
        """
        Determines the mode of the distribution, i.e., the point where the pdf is largest.
        Returns:
            m (lin_dim + bound_dim,) vector: the mode
        """
        return self.mu

    def set_mode(self, new_mode):
        self.mu = copy.copy(new_mode)
        return self

    def hybrid_moment(self):
        """
        Calculates mean of [x1, x2, .., x_lin_dim, cos(x_(linD+1), sin(x_(linD+1)), ..., cos(x_(linD+boundD), sin(x_(lin_dim+bound_dim))]
        Returns:
            mu (linD+2): expectation value of [x1, x2, .., x_lin_dim, cos(x_(lin_dim+1), sin(x_(lin_dim+1)), ..., cos(x_(lin_dim+bound_dim), sin(x_(lin_dim+bound_dim))]
        """
        mu_lin = self.mu[self.bound_dim :]  # noqa: E203

        mu_bound_odd = sin(self.mu[: self.bound_dim]) * exp(
            -diag(self.C)[: self.bound_dim] / 2
        )
        mu_bound_even = cos(self.mu[: self.bound_dim]) * exp(
            -diag(self.C)[: self.bound_dim] / 2
        )

        mu_bound = stack([mu_bound_even, mu_bound_odd], axis=1).reshape(-1)

        return hstack((mu_bound, mu_lin))

    def hybrid_mean(self):
        return self.mu

    def linear_mean(self):
        return self.mu[-self.lin_dim :]  # noqa: E203

    def periodic_mean(self):
        return self.mu[: self.bound_dim]

    def sample(self, n: int):
        """
        Sample n points from the distribution
        Parameters:
            n (int): number of points to sample
        """
        assert n > 0, "n must be positive"
        s = random.multivariate_normal(mean=self.mu, cov=self.C, size=(n,))
        wrapped_values = mod(s[:, : self.bound_dim], 2.0 * pi)
        unbounded_values = s[:, self.bound_dim :]  # noqa: E203

        # Concatenate the modified section with the unmodified section
        s = concatenate([wrapped_values, unbounded_values], axis=1)
        return s

    def to_gaussian(self):
        return GaussianDistribution(self.mu, self.C)

    def linear_covariance(self):
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
