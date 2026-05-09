import warnings

from beartype import beartype

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    abs,
    all,
    allclose,
    argmax,
    argmin,
    concatenate,
    linalg,
    minimum,
    vstack,
)

from ...sampling.hyperspherical_sampler import get_grid_hyperhemisphere
from .abstract_hyperhemispherical_distribution import (
    AbstractHyperhemisphericalDistribution,
)
from .abstract_hypersphere_subset_distribution import (
    AbstractHypersphereSubsetDistribution,
)
from .abstract_hypersphere_subset_grid_distribution import (
    AbstractHypersphereSubsetGridDistribution,
)
from .custom_hyperhemispherical_distribution import CustomHyperhemisphericalDistribution
from .hyperspherical_dirac_distribution import HypersphericalDiracDistribution


class HyperhemisphericalGridDistribution(
    AbstractHypersphereSubsetGridDistribution, AbstractHyperhemisphericalDistribution
):

    def __init__(self, grid, grid_values, enforce_pdf_nonnegative=True):
        # Do not test norm precisely, only quick test if any coordinate exceeds 1.
        assert all(
            abs(grid) <= 1 + 1e-12
        ), "Grid points must not lie outside the unit cube."
        assert all(
            grid[:, -1] >= -1e-12
        ), "Always using upper hemisphere (along last dimension)."

        super().__init__(grid, grid_values, enforce_pdf_nonnegative)

    # ------------------------------------------------------------------
    # Basic functionality
    # ------------------------------------------------------------------
    def mean_direction(self):
        """
        For hyperhemispheres this method returns the *mode* (grid point with
        maximum weight) rather than the true mean direction.

        Using the spherical mean would bias the result toward [0; ...; 0; 1], since the lower
        half of the sphere is not represented.
        """
        warnings.warn(
            "For hyperhemispheres, `mean_direction` returns the mode and "
            "not the true mean direction.",
            UserWarning,
        )
        index_max = argmax(self.grid_values)
        mu = self.grid[index_max, :]
        return mu

    def to_full_sphere(self, method="antipodal"):
        """
        Convert hemisphere to full sphere.

        The grid is mirrored, and the values are halved to keep the resulting
        hyperspherical distribution normalized.
        """
        assert method == "antipodal", (
            "Currently only 'antipodal' method is supported for "
            "converting hyperhemispherical grid distributions to "
            "full-sphere grid distributions."
        )
        from .hyperspherical_grid_distribution import HypersphericalGridDistribution

        grid_full = vstack((self.grid, -self.grid))
        grid_values_ = 0.5 * concatenate((self.grid_values, self.grid_values))
        hgd = HypersphericalGridDistribution(grid_full, grid_values_)
        return hgd

    def plot(self):
        hdd = HypersphericalDiracDistribution(self.grid, self.grid_values.T)
        h = hdd.plot()
        return h

    def plot_interpolated(self):
        hdgd = self.to_full_sphere()

        def pdf_doubled(x):
            return 2 * hdgd.pdf(x)

        hhgd_interp = CustomHyperhemisphericalDistribution(pdf_doubled, 3)
        h = hhgd_interp.plot()
        return h

    # ------------------------------------------------------------------
    # Grid geometry utilities
    # ------------------------------------------------------------------
    def get_closest_point(self, xs):
        """
        Return the closest grid point(s) on the hemisphere, taking the symmetry
        x ~ -x into account.

        Parameters
        ----------
        xs : array_like
            Either a single point of shape (dim,),
            or an array of shape (n_points, dim).

        Returns
        -------
        points : ndarray
            Closest grid point(s), shape (dim,).
        indices : ndarray or int
            Indices of the closest grid points.
        """

        if xs.ndim == 1:
            if xs.shape[0] != self.dim:
                raise ValueError(f"xs must have length {self.dim}, got {xs.shape[0]}.")
            xs = xs[None, :]  # (1, dim)
        elif xs.ndim == 2:
            if xs.shape[1] == self.dim:
                pass  # already (batch, dim)
            elif xs.shape[0] == self.dim:
                xs = xs.T  # (batch, dim)
            else:
                raise ValueError(f"xs must have shape (n, dim) with dim={self.dim}.")
        else:
            raise ValueError("xs must be a 1D or 2D array.")

        # Distances to each grid point and its antipode.
        diff1 = xs[:, None, :] - self.grid[None, :, :]  # (batch, n_grid, dim)
        diff2 = xs[:, None, :] + self.grid[None, :, :]  # (batch, n_grid, dim)

        dists1 = linalg.norm(diff1, axis=2)  # (batch, n_grid)
        dists2 = linalg.norm(diff2, axis=2)  # (batch, n_grid)

        all_distances = minimum(dists1, dists2)  # (batch, n_grid)

        indices = argmin(all_distances, axis=1)  # (batch,)
        points = self.get_grid_point(indices)

        # For a single query, return 1D outputs for convenience.
        if points.ndim == 2 and points.shape[0] == 1:
            points = points[0, :]
            indices = indices[0]

        return points, indices

    def get_manifold_size(self):
        return 0.5 * (
            AbstractHypersphereSubsetDistribution.compute_unit_hypersphere_surface(
                self.grid.shape[1] - 1
            )
        )

    # ------------------------------------------------------------------
    # Multiplication on the hemisphere
    # ------------------------------------------------------------------
    @beartype
    def multiply(
        self: "HyperhemisphericalGridDistribution",
        other: "HyperhemisphericalGridDistribution",
    ) -> "HyperhemisphericalGridDistribution":
        """
        Multiply two hyperhemispherical grid distributions that share the same grid.

          1. Convert both to full-sphere grid distributions.
          2. Multiply them as HypersphericalGridDistribution objects.
          3. Restrict back to the hemisphere and rescale.

        Parameters
        ----------
        other : HyperhemisphericalGridDistribution

        Returns
        -------
        HyperhemisphericalGridDistribution

        Raises
        ------
        ValueError
            If the grids are not identical (up to numerical tolerance).
        """
        if (
            self.dim != other.dim
            or self.get_grid().shape != other.get_grid().shape
            or not allclose(self.get_grid(), other.get_grid())
        ):
            raise ValueError("Multiply:IncompatibleGrid")

        # 1–2. Multiply on the full sphere using the full-sphere implementation.
        hgd1 = self.to_full_sphere()
        hgd2 = other.to_full_sphere()
        hgd_filtered = hgd1.multiply(hgd2)

        # 3. Restrict to hemisphere and rescale:
        n_hemi = self.grid.shape[0]
        hemi_grid = hgd_filtered.grid[:n_hemi]
        hemi_values = 2.0 * hgd_filtered.grid_values[:n_hemi]

        return HyperhemisphericalGridDistribution(
            hemi_grid, hemi_values, enforce_pdf_nonnegative=True
        )

    # ------------------------------------------------------------------
    # Construction from a function handle
    # ------------------------------------------------------------------
    # pylint: disable=too-many-locals
    @staticmethod
    def from_function(
        fun,
        no_of_grid_points,
        dim=2,
        grid_type="leopardi_symm",
        enforce_pdf_nonnegative=True,
    ):
        """
        Construct a hyperhemispherical grid distribution from a callable.

        Parameters
        ----------
        fun : callable
            A function mapping an array of shape (batch_dim, space_dim)
            to a 1-D array of pdf values.  In particular this matches
            your Python convention `pdf(x)` with x.shape == (batch, dim).
        no_of_grid_points : int
            Number of grid points on the hemisphere.
        dim : int, optional
            Ambient dimension (space_dim). Default is 2 for S².
        grid_type : {'leopardi_symm', 'healpix'}

        Notes
        -----
        For 'leopardi*' types, the grid is deterministic and only
        depends on (dim, no_of_grid_points, grid_type). For 'healpix',
        only dim == 3 is supported and `healpy` must be installed.
        """
        assert grid_type in (
            "leopardi_symm",
            "healpix",
        ), "For hyperhemispheres, use one of the symmetric grid types 'leopardi_symm' or 'healpix'."
        grid, _ = get_grid_hyperhemisphere(grid_type, no_of_grid_points, dim=dim)

        grid_values = fun(grid)

        sgd = HyperhemisphericalGridDistribution(
            grid, grid_values, enforce_pdf_nonnegative=enforce_pdf_nonnegative
        )
        sgd.grid_type = grid_type
        return sgd
