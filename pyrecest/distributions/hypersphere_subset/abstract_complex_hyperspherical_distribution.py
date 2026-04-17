import math
from abc import abstractmethod

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import pi
from pyrecest.distributions.abstract_manifold_specific_distribution import (
    AbstractManifoldSpecificDistribution,
)


class AbstractComplexHypersphericalDistribution(AbstractManifoldSpecificDistribution):
    """
    Abstract base class for distributions on the complex unit hypersphere in C^d.

    The complex unit hypersphere in C^d is the set
        {z in C^d : ||z|| = 1}
    which is isomorphic to the real sphere S^{2d-1} in R^{2d}.

    The complex dimension d is stored as ``complex_dim``.
    The underlying manifold dimension (as a real manifold) is 2*d - 1.
    """

    def __init__(self, complex_dim: int):
        """
        Parameters
        ----------
        complex_dim : int
            Complex dimension d of the ambient space C^d (d >= 1).
        """
        if complex_dim < 1:
            raise ValueError("complex_dim must be >= 1.")
        # The real manifold dimension is 2*d - 1
        super().__init__(2 * complex_dim - 1)
        self._complex_dim = complex_dim

    @property
    def complex_dim(self) -> int:
        """Complex dimension d of the ambient space C^d."""
        return self._complex_dim

    @property
    def input_dim(self) -> int:
        """Number of complex coordinates of a point on the sphere (= complex_dim)."""
        return self._complex_dim

    def get_manifold_size(self) -> float:
        """Surface area of the real sphere S^{2d-1} = 2*pi^d / (d-1)!"""
        d = self._complex_dim
        return float(2 * pi**d / math.factorial(d - 1))

    @abstractmethod
    def pdf(self, xs):
        """Probability density at the given point(s) on the complex unit sphere."""

    def mean(self):
        raise NotImplementedError(
            "mean() is not defined for this complex hyperspherical distribution."
        )
