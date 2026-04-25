# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import linalg, log, outer, reshape, sum, zeros

from ..abstract_dirac_distribution import AbstractDiracDistribution
from .abstract_hypersphere_subset_distribution import (
    AbstractHypersphereSubsetDistribution,
)


class AbstractHypersphereSubsetDiracDistribution(
    AbstractDiracDistribution, AbstractHypersphereSubsetDistribution
):
    def __init__(self, d, w=None):
        AbstractHypersphereSubsetDistribution.__init__(self, d.shape[-1] - 1)
        AbstractDiracDistribution.__init__(self, d, w=w)

    def moment(self):
        # Compute the weighted moment matrix
        moment_matrix = zeros(
            (self.d.shape[1], self.d.shape[1])
        )  # Initialize (dim, dim) matrix
        for i in range(self.d.shape[0]):  # Iterate over samples
            moment_matrix += self.w[i] * outer(self.d[i, :], self.d[i, :])

        return moment_matrix

    def entropy(self):
        result = -sum(self.w * log(self.w))
        return result

    def integrate(self, left=None, right=None):
        _ = left, right
        raise NotImplementedError()

    def mean_axis(self):
        """
        Returns the principal axis of the Dirac mixture on the hypersphere.
        Because ±v represent the same axis, the sign of the returned vector
        is arbitrary.
        """
        # Column vector of weights, shape (n, 1)
        w_col = reshape(self.w, (-1, 1))  # or self.w[:, None]

        # Weighted second-moment matrix: S = Σ w_i d_i d_i^T
        # d has shape (n, D), so (d * w_col) is (n, D), then transpose @ d -> (D, D)
        S = (self.d * w_col).T @ self.d

        # Normalize in case weights don't sum to 1
        S = S / sum(self.w)

        # Enforce symmetry (numerical safety)
        S = 0.5 * (S + S.T)

        # Eigen-decomposition of symmetric S
        D, V = linalg.eigh(S)

        # Index of largest eigenvalue
        # If you don't have argmax in the backend, use argsort instead
        idx = D.argmax()  # or idx = argsort(D)[-1]

        axis = V[:, idx]

        # Normalize to unit length (should already be, but just in case)
        axis = axis / linalg.norm(axis)

        return axis
