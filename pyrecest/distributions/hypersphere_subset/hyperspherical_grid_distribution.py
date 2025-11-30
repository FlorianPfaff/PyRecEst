import warnings

from ...sampling.hyperspherical_sampler import get_grid_hypersphere

from .abstract_hypersphere_subset_grid_distribution import (
    AbstractHypersphereSubsetGridDistribution,
)
from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
from .hyperhemispherical_grid_distribution import HyperhemisphericalGridDistribution

import copy

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import abs, all, linalg, concatenate, allclose, argmax, argmin

class HypersphericalGridDistribution(
    AbstractHypersphereSubsetGridDistribution, AbstractHypersphericalDistribution
):
    """
    Convention:
    - `self.grid` is shape (n_points, input_dim)
    - `self.grid_values` is shape (n_points,)
    - `pdf(x)` expects x of shape (batch_dim, input_dim)
    """

    def __init__(
        self,
        grid,
        grid_values_,
        enforce_pdf_nonnegative=True,
        grid_type="unknown",
    ):

        if grid.ndim != 2:
            raise ValueError("grid_ must be a 2D array of shape (n_points, input_dim).")

        if grid.shape[0] != grid_values_.shape[0]:
            raise ValueError(
                "grid_values_ must have length equal to the number of grid points "
                "(rows of grid_)."
            )

        if not all(abs(grid) <= 1 + 1e-12):
            raise ValueError(
                "Grid points must not lie outside the unit square (otherwise they are outside the domain)"
                "(-1 <= coordinates <= 1)."
            )

        AbstractHypersphereSubsetGridDistribution.__init__(self, grid, grid_values_, enforce_pdf_nonnegative)
        AbstractHypersphericalDistribution.__init__(self, grid.shape[1])
        self.grid_type = grid_type

    def get_manifold_size(self):
        return AbstractHypersphericalDistribution.get_manifold_size(self)

    # ------------------------------------------------------------------
    # Basic statistics
    # ------------------------------------------------------------------
    def mean_direction(self):
        """
        Mean direction on the hypersphere.
        """
        mu = (self.grid.T @ self.grid_values).reshape((-1,))  # (dim,)
        norm_mu = linalg.norm(mu)

        if norm_mu < 1e-8:
            warnings.warn(
                "Density may not actually have a mean direction because "
                "formula yields a point very close to the origin.",
                UserWarning,
            )
            if norm_mu == 0.0:
                return mu

        return mu / norm_mu

    # ------------------------------------------------------------------
    # PDF (nearest-neighbour / piecewise constant interpolation)
    # ------------------------------------------------------------------
    def pdf(self, xs):
        """
        Piecewise-constant interpolated pdf.

        xs can be:
        - shape (dim,)
        - shape (batch, dim)

        Returns:
        - scalar if input is 1D
        - (batch,) if input is 2D
        """
        warnings.warn(
            "PDF:UseInterpolated: Interpolating the pdf with constant values in each "
            "region is not very efficient, but it is good enough for "
            "visualization purposes.",
            UserWarning,
        )

        # scores: (n_grid, batch)
        scores = self.grid @ xs.T
        max_indices = argmax(scores, axis=0)  # (batch,)

        vals = self.grid_values[max_indices]  # (batch,)

        return vals

    # ------------------------------------------------------------------
    # Symmetrization & hemisphere operations
    # ------------------------------------------------------------------
    def symmetrize(self):
        """
        Make the grid distribution antipodally symmetric.

        Requires a symmetric grid: the second half of the grid is the negation
        of the first half.

        New grid_values are the average of each pair, copied to both points.
        """
        n = self.grid.shape[0]
        if n % 2 != 0:
            raise ValueError(
                "Symmetrize:AsymmetricGrid: grid must have an even number of points."
            )

        half = n // 2
        if not allclose(self.grid[:half], -self.grid[half:], atol=1e-12):
            raise ValueError(
                "Symmetrize:AsymmetricGrid: "
                "Can only use symmetrize for symmetric grids. "
                "Use grid_type 'eq_point_set_symm' when calling from_distribution "
                "or from_function."
            )

        grid_values_half = 0.5 * (
            self.grid_values[:half] + self.grid_values[half:]
        )
        new_values = concatenate([grid_values_half, grid_values_half])

        return HypersphericalGridDistribution(
            copy.deepcopy(self.grid), new_values, enforce_pdf_nonnegative=True, grid_type=self.grid_type
        )

    def to_hemisphere(self, tol=1e-10):
        """
        Convert a symmetric full-sphere grid distribution to a
        HyperhemisphericalGridDistribution on the upper hemisphere.

        If the density appears asymmetric (pairwise grid values differ by
        more than `tol`), the hemisphere values are formed by summing
        symmetric pairs instead of 2 * first_half.
        """
        n = self.grid.shape[0]
        if n % 2 != 0:
            raise ValueError(
                "ToHemisphere:AsymmetricGrid: grid must have an even number of points."
            )

        half = n // 2
        if not allclose(self.grid[:half, :], -self.grid[half:, :], atol=1e-12):
            raise ValueError(
                "ToHemisphere:AsymmetricGrid: "
                "Can only use to_hemisphere for symmetric grids. "
                "Use grid_type 'eq_point_set_symm' when calling from_distribution "
                "or from_function."
            )

        if allclose(self.grid_values[:half], self.grid_values[half:], atol=tol):
            grid_values_hemisphere = 2.0 * self.grid_values[:half]
        else:
            warnings.warn(
                "ToHemisphere:AsymmetricDensity: Density appears to be asymmetric. "
                "Using sum of symmetric pairs instead of 2*first_half.",
                UserWarning,
            )
            grid_values_hemisphere = self.grid_values[:half] + self.grid_values[half:]

        hemi_grid = self.grid[:half]
        return HyperhemisphericalGridDistribution(hemi_grid, grid_values_hemisphere)

    # ------------------------------------------------------------------
    # Geometry: closest grid point
    # ------------------------------------------------------------------
    def get_closest_point(self, xs):
        """
        Return closest grid point(s) in Euclidean distance.

        xs can be:
        - shape (dim,)
        - shape (batch, dim)

        Returns
        -------
        points : ndarray
            Shape (dim,) for single query or (batch, dim) for multiple.
        indices : int or ndarray
            Index/indices of closest grid points.
        """
        if xs.ndim == 1:
            if xs.shape[0] != self.dim:
                raise ValueError(
                    f"Expected xs of length {self.dim}, got {xs.shape[0]}."
                )
            xs = xs[None, :]  # (1, dim)
        elif xs.ndim == 2:
            assert xs.shape[-1] == self.dim
        else:
            raise ValueError("xs must be 1D or 2D array.")

        diff = xs[:, None, :] - self.grid[None, :, :]  # (batch, n_grid, dim)
        dists = linalg.norm(diff, axis=2)  # (batch, n_grid)
        indices = argmin(dists, axis=1)  # (batch,)
        points = self.get_grid_point(indices)

        return points, indices

    # ------------------------------------------------------------------
    # Multiply (with compatibility check)
    # ------------------------------------------------------------------
    def multiply(self, other):
        """
        Multiply two hyperspherical grid distributions defined on the same grid.

        This method simply checks grid compatibility and then delegates to the
        superclass multiply implementation.
        """
        if not isinstance(other, HypersphericalGridDistribution):
            return super().multiply(other)

        if (
            self.dim != other.dim
            or self.grid.shape != other.grid.shape
            or not allclose(self.grid, other.grid, atol=1e-12)
        ):
            raise ValueError("Multiply:IncompatibleGrid")

        return super().multiply(other)

    # ------------------------------------------------------------------
    # Construction from other distributions
    # ------------------------------------------------------------------
    @staticmethod
    def from_distribution(
        distribution,
        no_of_grid_points,
        grid_type="eq_point_set",
        enforce_pdf_nonnegative=True,
    ):
        """
        Approximate an AbstractHypersphericalDistribution on a grid.
        """
        if not isinstance(distribution, AbstractHypersphericalDistribution):
            raise TypeError(
                "distribution must be an instance of AbstractHypersphericalDistribution."
            )

        fun = distribution.pdf
        return HypersphericalGridDistribution.from_function(
            fun, no_of_grid_points, distribution.dim, grid_type, enforce_pdf_nonnegative
        )

    @staticmethod
    def from_function(
        fun, no_of_grid_points, dim, grid_type="eq_point_set", enforce_pdf_nonnegative=True
    ):
        """
        Construct a HypersphericalGridDistribution from a callable.

        Parameters
        ----------
        fun : callable
            Function taking an array of shape (batch_dim, space_dim) and
            returning a 1D array of pdf values.
        no_of_grid_points : int
            Grid parameter (interpreted as number of points for 'eq_point_set'
            and total number of points for symmetric schemes).
        dim : int
            Ambient space dimension (>= 2).
        grid_type : str
            Type of grid to use. See `get_grid_hypersphere` for options.
        enforce_pdf_nonnegative : bool
            Whether to enforce non-negativity of grid values in base class.
        """
        if dim < 2:
            raise ValueError("dim must be >= 2")

        grid, _ = get_grid_hypersphere(grid_type, no_of_grid_points, dim)
        # Call user pdf with X of shape (batch_dim, space_dim) = (n_points, dim)
        grid_values = fun(grid)

        return HypersphericalGridDistribution(
            grid, grid_values, enforce_pdf_nonnegative=enforce_pdf_nonnegative, grid_type=grid_type
        )
