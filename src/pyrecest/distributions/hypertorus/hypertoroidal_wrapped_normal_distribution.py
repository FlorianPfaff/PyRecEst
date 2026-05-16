import copy
from typing import Union

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    allclose,
    arange,
    array,
    atleast_2d,
    exp,
    int32,
    int64,
    linalg,
    meshgrid,
    mod,
    pi,
    random,
    stack,
    zeros,
)
from scipy.stats import multivariate_normal

from .abstract_hypertoroidal_distribution import AbstractHypertoroidalDistribution


class HypertoroidalWrappedNormalDistribution(AbstractHypertoroidalDistribution):
    def __init__(self, mu, C):
        """
        Initialize HypertoroidalWrappedNormalDistribution.

        :param mu: Mean vector.
        :param C: Covariance matrix.
        :raises AssertionError: If C_ is not square, not symmetric, not positive definite, or its dimension does not match with mu_.
        """
        numel_mu = 1 if mu.ndim == 0 else mu.shape[0]
        assert (
            C.ndim == 0 or C.ndim == 2 and C.shape[0] == C.shape[1]
        ), "C must be of shape (dim, dim)"
        assert C.ndim == 0 or allclose(C, C.T, atol=1e-8), "C must be symmetric"
        assert (
            C.ndim == 0
            and C > 0.0
            or len(linalg.cholesky(C)) > 0  # fails if not positiv definite
        ), "C must be positive definite"
        assert numel_mu == 1 or mu.shape == (C.shape[1],), "mu must be of shape (dim,)"
        AbstractHypertoroidalDistribution.__init__(self, numel_mu)
        self.mu = mod(mu, 2.0 * pi)
        self.C = C

    def set_mean(self, mu):
        """
        Set the mean of the distribution.

        Parameters:
        mu (numpy array): The new mean.

        Returns:
        HypertoroidalWNDistribution: A new instance of the distribution with the updated mean.
        """
        dist = copy.deepcopy(self)
        dist.mu = mod(mu, 2.0 * pi)
        return dist

    def pdf(self, xs, m: Union[int, int32, int64] = 3):
        """
        Compute the PDF at given points.

        :param xs: Points to evaluate the PDF at.
        :param m: Controls the number of terms in the Fourier series approximation.
        :return: PDF values at xs.
        """
        assert (
            xs.shape[-1] == self.dim
        ), "Last dimension of xs must match the distribution dimension"
        xs = atleast_2d(xs)  # Ensure xs is at least 2-D for approach below
        xs = (xs + pi) % (2 * pi) - pi

        # Generate all combinations of offsets for each dimension
        offsets = [arange(-m, m + 1) * 2.0 * pi for _ in range(self.dim)]
        offset_combinations = stack(meshgrid(*offsets, indexing="ij"), -1).reshape(
            -1, self.dim
        )

        # Calculate the PDF values by considering all combinations of offsets
        pdf_values = zeros(xs.shape[0])
        for offset in offset_combinations:
            shifted_xa = xs + offset[None, :]
            pdf_values += multivariate_normal.pdf(
                shifted_xa, mean=self.mu.flatten(), cov=self.C
            )

        return pdf_values

    def shift(self, shift_by) -> "HypertoroidalWrappedNormalDistribution":
        """
        Shift distribution by the given angles

        :param shift_by: Angles to shift by.
        :raises AssertionError: If shape of shift_by does not match the dimension of the distribution.
        :return: Shifted distribution.
        """
        assert shift_by.shape == (self.dim,)

        return self.set_mean(self.mu + shift_by)

    def sample(self, n: Union[int, int32, int64]):
        if n <= 0:
            raise ValueError("n must be a positive integer")

        s = random.multivariate_normal(self.mu, self.C, (n,))
        s = mod(s, 2.0 * pi)  # wrap the samples
        return s

    def convolve(self, other: "HypertoroidalWrappedNormalDistribution"):
        assert self.dim == other.dim, "Dimensions of the two distributions must match"
        mu_ = (self.mu + other.mu) % (2.0 * pi)
        C_ = self.C + other.C
        dist_result = self.__class__(mu_, C_)
        return dist_result

    def set_mode(self, m):
        """
        Set the mode of the distribution.

        Parameters:
        m (numpy array): The new mode.

        Returns:
        HypertoroidalWNDistribution: A new instance of the distribution with the updated mode.
        """
        dist = copy.deepcopy(self)
        dist.mu = m
        return dist

    def trigonometric_moment(self, n):
        """
        Calculate the trigonometric moment of the HypertoroidalWNDistribution.

        :param self: HypertoroidalWNDistribution instance
        :param n: Integer moment order
        :return: Trigonometric moment
        """
        assert isinstance(n, int), "n must be an integer"

        m = exp(
            array(
                [1j * n * self.mu[i] - n**2 * self.C[i, i] / 2 for i in range(self.dim)]
            )
        )

        return m

    def mode(self):
        # Determines the mode of the distribution, i.e., the point
        # where the pdf is largest.
        #
        # Returns:
        #   m (vector)
        #       the mode
        return self.mu
