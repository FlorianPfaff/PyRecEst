# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, diff, prod, reshape, to_numpy
from scipy.integrate import nquad

from ..abstract_bounded_nonperiodic_distribution import (
    AbstractBoundedNonPeriodicDistribution,
)


class AbstractHyperrectangularDistribution(AbstractBoundedNonPeriodicDistribution):
    def __init__(self, bounds):
        bounds = array(bounds)
        if bounds.ndim == 1:
            if bounds.shape[0] != 2:
                raise ValueError("one-dimensional bounds must have length 2")
            bounds = reshape(bounds, (1, 2))
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (dim, 2)")
        AbstractBoundedNonPeriodicDistribution.__init__(self, int(bounds.shape[0]))
        self.bounds = bounds

    def get_manifold_size(self):
        s = prod(diff(self.bounds, axis=1))
        return s

    @property
    def input_dim(self):
        return self.dim

    def integrate(self, integration_boundaries=None) -> float:
        """
        Integrate the probability density function over given boundaries.
        If no boundaries are provided, default to `self.bounds`.

        Args:
            integration_boundaries (tuple): A tuple of two elements, each of which can be either a scalar or an array.
                If a scalar, it represents a single boundary value.
                If an array, it represents multiple boundary values.

        Returns:
            float: The result of the integration.
        """
        if integration_boundaries is None:
            integration_boundaries = self.bounds

        integration_boundaries = reshape(array(integration_boundaries), (-1, 2))
        if integration_boundaries.shape[0] != self.dim:
            raise ValueError(f"integration_boundaries must have shape ({self.dim}, 2)")
        left = integration_boundaries[:, 0]
        right = integration_boundaries[:, 1]
        ranges = [
            (float(lower), float(upper))
            for lower, upper in zip(to_numpy(left), to_numpy(right))
        ]

        def integrand(*args):
            values = self.pdf(reshape(array(args), (1, self.dim)))
            return float(to_numpy(array(values)).reshape(-1)[0])

        return nquad(integrand, ranges)[0]
