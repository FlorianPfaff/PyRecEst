import numpy as np
import warnings

from .abstract_hypersphere_subset_grid_distribution import (
    AbstractHypersphereSubsetGridDistribution,
)
from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
from .hyperspherical_dirac_distribution import HypersphericalDiracDistribution
from .custom_hyperspherical_distribution import CustomHypersphericalDistribution
from .hyperhemispherical_grid_distribution import HyperhemisphericalGridDistribution


class HypersphericalGridDistribution(
    AbstractHypersphereSubsetGridDistribution, AbstractHypersphericalDistribution
):
    """
    Hyperspherical grid distribution (full sphere), Python port of the
    MATLAB class HypersphericalGridDistribution.

    Internal convention:
    - `self.grid` is shape (dim, n_points)
    - `self.grid_values` is shape (n_points,)
    - `pdf(x)` expects x of shape (batch_dim, space_dim)
      (i.e. (N, dim) in your Python convention)
    """

    def __init__(
        self,
        grid_,
        grid_values_,
        enforce_pdf_nonnegative=True,
        grid_type="unknown",
    ):
        grid_ = np.asarray(grid_, dtype=float)
        grid_values_ = np.asarray(grid_values_, dtype=float).reshape(-1)

        if grid_.ndim != 2:
            raise ValueError("grid_ must be a 2D array of shape (dim, n_points).")

        if grid_.shape[1] != grid_values_.shape[0]:
            raise ValueError(
                "grid_values_ must have length equal to the number of grid points "
                "(columns of grid_)."
            )

        if not np.all(np.abs(grid_) <= 1 + 1e-12):
            raise ValueError(
                "Grid points must lie on or inside the unit hypersphere "
                "(-1 <= coordinates <= 1)."
            )

        super().__init__(grid_, grid_values_, enforce_pdf_nonnegative)
        self.grid_type = grid_type

    # ------------------------------------------------------------------
    # Basic statistics
    # ------------------------------------------------------------------
    def mean_direction(self):
        """
        Mean direction on the hypersphere, analogous to MATLAB meanDirection.

        mu = sum_j grid[:, j] * grid_values[j]
        """
        mu = self.grid @ self.grid_values  # (dim,)
        norm_mu = np.linalg.norm(mu)

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
        - shape (dim, batch)

        Returns:
        - scalar if input is 1D
        - (batch,) if input is 2D
        """
        xs = np.asarray(xs, dtype=float)

        warnings.warn(
            "PDF:UseInterpolated: Interpolating the pdf with constant values in each "
            "region is not very efficient, but it is good enough for "
            "visualization purposes.",
            UserWarning,
        )

        # Normalize shapes to (batch, dim)
        if xs.ndim == 1:
            if xs.shape[0] != self.dim:
                raise ValueError(
                    f"Expected 1D xs of length {self.dim}, got {xs.shape[0]}."
                )
            xs = xs[None, :]  # (1, dim)
            single = True
        elif xs.ndim == 2:
            if xs.shape[1] == self.dim and xs.shape[0] != self.dim:
                single = False  # already (batch, dim)
            elif xs.shape[0] == self.dim and xs.shape[1] != self.dim:
                xs = xs.T  # (batch, dim)
                single = False
            elif xs.shape[0] == self.dim and xs.shape[1] == self.dim:
                # ambiguous; treat as (batch, dim)
                single = False
            else:
                raise ValueError(
                    f"xs must have shape (dim,), (batch, dim), or (dim, batch) with "
                    f"dim={self.dim}, got {xs.shape}."
                )
        else:
            raise ValueError("xs must be 1D or 2D array.")

        # self.grid: (dim, n_grid)
        grid_T = self.grid.T  # (n_grid, dim)
        # scores: (n_grid, batch)
        scores = grid_T @ xs.T
        max_indices = np.argmax(scores, axis=0)  # (batch,)

        vals = self.grid_values[max_indices]  # (batch,)

        if single:
            return float(vals[0])
        return vals

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot(self):
        """
        Simple plot using a dirac mixture on the grid.

        For S^2 (dim=3), if your AbstractHypersphericalDistribution has a
        helper to draw the sphere, you can plug it in here.
        """
        if self.dim == 3:
            # If such a helper exists, you might want:
            # AbstractHypersphericalDistribution.plot_sphere()
            pass

        # Normalize weights for plotting
        weights = self.grid_values / np.sum(self.grid_values)
        hdd = HypersphericalDiracDistribution(self.grid, weights.T)
        return hdd.plot()

    def plot_interpolated(self):
        """
        Plot an interpolated version of the grid pdf using a
        CustomHypersphericalDistribution wrapper.
        """
        chd = CustomHypersphericalDistribution(lambda x: self.pdf(x), self.dim)
        return chd.plot()

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
        n = self.grid.shape[1]
        if n % 2 != 0:
            raise ValueError(
                "Symmetrize:AsymmetricGrid: grid must have an even number of points."
            )

        half = n // 2
        if not np.allclose(self.grid[:, :half], -self.grid[:, half:], atol=1e-12):
            raise ValueError(
                "Symmetrize:AsymmetricGrid: "
                "Can only use symmetrize for symmetric grids. "
                "Use grid_type 'eq_point_set_symm' when calling from_distribution "
                "or from_function."
            )

        grid_values_half = 0.5 * (
            self.grid_values[:half] + self.grid_values[half:]
        )
        new_values = np.concatenate([grid_values_half, grid_values_half])

        return HypersphericalGridDistribution(
            self.grid.copy(), new_values, enforce_pdf_nonnegative=True, grid_type=self.grid_type
        )

    def to_hemisphere(self, tol=1e-10):
        """
        Convert a symmetric full-sphere grid distribution to a
        HyperhemisphericalGridDistribution on the upper hemisphere.

        If the density appears asymmetric (pairwise grid values differ by
        more than `tol`), the hemisphere values are formed by summing
        symmetric pairs instead of 2 * first_half.
        """
        n = self.grid.shape[1]
        if n % 2 != 0:
            raise ValueError(
                "ToHemisphere:AsymmetricGrid: grid must have an even number of points."
            )

        half = n // 2
        if not np.allclose(self.grid[:, :half], -self.grid[:, half:], atol=1e-12):
            raise ValueError(
                "ToHemisphere:AsymmetricGrid: "
                "Can only use to_hemisphere for symmetric grids. "
                "Use grid_type 'eq_point_set_symm' when calling from_distribution "
                "or from_function."
            )

        first_half = self.grid_values[:half]
        second_half = self.grid_values[half:]

        if np.allclose(first_half, second_half, atol=tol):
            grid_values_hemisphere = 2.0 * first_half
        else:
            warnings.warn(
                "ToHemisphere:AsymmetricDensity: Density appears to be asymmetric. "
                "Using sum of symmetric pairs instead of 2*first_half.",
                UserWarning,
            )
            grid_values_hemisphere = first_half + second_half

        hemi_grid = self.grid[:, :half]
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
        - shape (dim, batch)

        Returns
        -------
        points : ndarray
            Shape (dim,) for single query or (dim, batch) for multiple.
        indices : int or ndarray
            Index/indices of closest grid points.
        """
        xs = np.asarray(xs, dtype=float)
        single = xs.ndim == 1

        if xs.ndim == 1:
            if xs.shape[0] != self.dim:
                raise ValueError(
                    f"Expected xs of length {self.dim}, got {xs.shape[0]}."
                )
            xs = xs[None, :]  # (1, dim)
        elif xs.ndim == 2:
            if xs.shape[1] == self.dim and xs.shape[0] != self.dim:
                pass  # (batch, dim)
            elif xs.shape[0] == self.dim and xs.shape[1] != self.dim:
                xs = xs.T  # (batch, dim)
            elif xs.shape[0] == self.dim and xs.shape[1] == self.dim:
                pass
            else:
                raise ValueError(
                    f"xs must have shape (dim,), (batch, dim) or (dim, batch) with "
                    f"dim={self.dim}, got {xs.shape}."
                )
        else:
            raise ValueError("xs must be 1D or 2D array.")

        grid_T = self.grid.T  # (n_grid, dim)
        diff = xs[:, None, :] - grid_T[None, :, :]  # (batch, n_grid, dim)
        dists = np.linalg.norm(diff, axis=2)  # (batch, n_grid)
        indices = np.argmin(dists, axis=1)  # (batch,)
        points = self.get_grid_point(indices)

        if single:
            return points[:, 0], int(indices[0])
        return points, indices

    # ------------------------------------------------------------------
    # Multiply (with compatibility check)
    # ------------------------------------------------------------------
    def multiply(self, other):
        """
        Multiply two hyperspherical grid distributions defined on the same grid.

        This method simply checks grid compatibility and then delegates to the
        superclass multiply implementation. If the grids are incompatible, a
        ValueError with message 'Multiply:IncompatibleGrid' is raised (to
        mirror MATLAB's error identifier used in the tests).
        """
        if not isinstance(other, HypersphericalGridDistribution):
            # Let the base class handle other types, if supported.
            return super().multiply(other)

        if (
            self.dim != other.dim
            or self.grid.shape != other.grid.shape
            or not np.allclose(self.grid, other.grid, atol=1e-12)
        ):
            raise ValueError("Multiply:IncompatibleGrid")

        return super().multiply(other)

    # ------------------------------------------------------------------
    # Static helpers: eq_point_set-style grids
    # ------------------------------------------------------------------
    @staticmethod
    def _uniform_points_on_sphere(dim, n_points, rng):
        """
        Sample n_points approximately uniformly from S^(dim-1).
        Returns an array of shape (n_points, dim).
        """
        pts = rng.normal(size=(n_points, dim))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts = pts / norms
        return pts

    @staticmethod
    def _eq_point_set_symm_hemisphere(dim, n_points, rng):
        """
        Hemisphere version of eq_point_set_symm.

        This is intentionally identical to the helper used in
        HyperhemisphericalGridDistribution so that:

        - HyperhemisphericalGridDistribution.from_function(..., 'eq_point_set_symm')
        - HypersphericalGridDistribution.from_function(..., 'eq_point_set_symm')

        generate compatible grids for the test that converts a hemisphere grid
        back to a full-sphere grid.
        """
        pts = rng.normal(size=(n_points, dim))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts = pts / norms

        # force points onto upper hemisphere (last coordinate >= 0)
        mask = pts[:, -1] < 0
        pts[mask] *= -1.0

        return pts.T  # (dim, n_points)

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
        grid_type : {'eq_point_set', 'eq_point_set_symm', 'eq_point_set_symmetric',
                     'eq_point_set_symm_plane'}
        enforce_pdf_nonnegative : bool
            Whether to enforce non-negativity of grid values in base class.
        """
        if dim < 2:
            raise ValueError("dim must be >= 2")

        no_of_grid_points = int(no_of_grid_points)

        if grid_type == "eq_point_set":
            seed = hash((dim, no_of_grid_points, grid_type)) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            pts = HypersphericalGridDistribution._uniform_points_on_sphere(
                dim, no_of_grid_points, rng
            )
            grid = pts.T  # (dim, n_points)

        elif grid_type in {"eq_point_set_symm", "eq_point_set_symmetric"}:
            if no_of_grid_points % 2 != 0:
                raise ValueError(
                    "eq_point_set_symm requires an even no_of_grid_points "
                    "(grid consists of antipodal pairs)."
                )
            n_hemi = no_of_grid_points // 2
            seed = hash((dim, n_hemi, grid_type)) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            hemi_grid = HypersphericalGridDistribution._eq_point_set_symm_hemisphere(
                dim, n_hemi, rng
            )  # (dim, n_hemi)
            grid = np.hstack((hemi_grid, -hemi_grid))  # (dim, 2 * n_hemi)

        elif grid_type == "eq_point_set_symm_plane":
            # Simple placeholder: treat it like a symmetric grid for now.
            if no_of_grid_points % 2 != 0:
                raise ValueError(
                    "eq_point_set_symm_plane requires an even no_of_grid_points."
                )
            n_hemi = no_of_grid_points // 2
            seed = hash((dim, n_hemi, grid_type)) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            hemi_grid = HypersphericalGridDistribution._eq_point_set_symm_hemisphere(
                dim, n_hemi, rng
            )
            grid = np.hstack((hemi_grid, -hemi_grid))

        else:
            raise ValueError("Grid scheme not recognized")

        # Call user pdf with X of shape (batch_dim, space_dim) = (n_points, dim)
        grid_values = np.asarray(fun(grid.T), dtype=float).reshape(-1)

        return HypersphericalGridDistribution(
            grid, grid_values, enforce_pdf_nonnegative=enforce_pdf_nonnegative, grid_type=grid_type
        )
