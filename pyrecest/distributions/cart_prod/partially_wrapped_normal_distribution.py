import copy
from typing import Union
from itertools import product
# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    allclose,
    arange,
    array,
    atleast_2d,
    concatenate,
    cos,
    diag,
    empty,
    exp,
    hstack,
    int32,
    int64,
    linalg,
    mod,
    ndim,
    pi,
    random,
    repeat,
    sin,
    stack,
    sum,
    where,
    zeros,
)
from scipy.stats import multivariate_normal

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

    # pylint: disable=too-many-locals
    def pdf(self, xs, m=3):
        """
        Evaluate the PDF of the Hypercylindrical Wrapped Normal Distribution at given points.

        Parameters:
            xs (array-like): Input points of shape (n, d), where d = bound_dim + lin_dim.
            m (int, optional): Number of summands in each direction for wrapping. Default is 3.

        Returns:
            p (ndarray): PDF values at each input point of shape (n,).
        """
        assert (
            pyrecest.backend.__backend_name__ == "numpy"
        ), "Only supported for numpy backend"

        xs = atleast_2d(xs)  # Ensure xs is 2D
        n, d = xs.shape
        assert (
            d == self.dim
        ), f"Input dimensionality {d} does not match distribution dimensionality {self.dim}."

        # Initialize the PDF values array
        p = zeros(n)

        # Define batch size to manage memory usage
        batch_size = 1000

        # Generate all possible offset combinations for periodic dimensions
        multiples = arange(-m, m + 1) * 2.0 * pi
        offset_combinations = list(
            product(multiples, repeat=self.bound_dim)
        )  # Total combinations: (2m+1)^bound_dim
        num_offsets = len(offset_combinations)

        # Pre-convert offset combinations to a NumPy array for efficient computation
        offset_array = array(offset_combinations)  # Shape: (num_offsets, bound_dim)

        # Process input data in batches
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = xs[start:end]  # Shape: (batch_size, d)

            # Wrap periodic dimensions using modulus
            batch_wrapped = batch.copy()
            if self.bound_dim > 0:
                batch_wrapped[:, : self.bound_dim] = mod(
                    batch_wrapped[:, : self.bound_dim], 2.0 * pi
                )  # noqa: E203

            if self.bound_dim > 0:
                # Correct broadcasting: batch_wrapped becomes (batch_size, 1, bound_dim)
                # offset_array becomes (1, num_offsets, bound_dim)
                wrapped_periodic = batch_wrapped[:, :self.bound_dim][:, None, :] + offset_array[None, :, :]
                # Now wrapped_periodic has shape (batch_size, num_offsets, bound_dim)
                wrapped_periodic = wrapped_periodic.reshape(-1, self.bound_dim)
            else:
                wrapped_periodic = empty((0, 0))  # No periodic dimensions

            # Repeat linear dimensions for each offset
            if self.lin_dim > 0:
                linear_part = repeat(
                    batch_wrapped[:, self.bound_dim :],  # noqa: E203
                    num_offsets,
                    axis=0,
                )  # Shape: (batch_size * num_offsets, lin_dim)
                # Concatenate wrapped periodic and linear parts
                if self.bound_dim > 0:
                    wrapped_points = hstack(
                        (wrapped_periodic, linear_part)
                    )  # Shape: (batch_size * num_offsets, d)
                else:
                    wrapped_points = linear_part  # Shape: (batch_size * num_offsets, d)
            else:
                wrapped_points = (
                    wrapped_periodic  # Shape: (batch_size * num_offsets, d)
                )

            mvn = multivariate_normal(mean=self.mu, cov=self.C)
            # Evaluate the multivariate normal PDF at all wrapped points
            pdf_vals = mvn.pdf(wrapped_points)  # Shape: (batch_size * num_offsets,)

            # Reshape and sum the PDF values for each original point
            pdf_vals = pdf_vals.reshape(
                end - start, num_offsets
            )  # Shape: (batch_size, num_offsets)
            p[start:end] = sum(pdf_vals, axis=1)  # Shape: (batch_size,)

        return p

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
        new_dist = copy.deepcopy(self)
        wrapped_mean = where(
            arange(new_mean.shape[0]) < self.bound_dim, mod(new_mean, 2 * pi), new_mean
        )
        new_dist.mu = wrapped_mean
        return new_dist

    def set_mode(self, new_mode):
        return self.set_mean(new_mode)

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
        wrapped_values = mod(s[:, : self.bound_dim], 2.0 * pi)  # noqa: E203
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
