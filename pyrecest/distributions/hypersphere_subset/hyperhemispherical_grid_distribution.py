from .abstract_hypersphere_subset_grid_distribution import (
    AbstractHypersphereSubsetGridDistribution,
)
from .abstract_hyperhemispherical_distribution import (
    AbstractHyperhemisphericalDistribution,
)
from .abstract_hyperspherical_distribution import AbstractHypersphericalDistribution
from .hyperspherical_grid_distribution import HypersphericalGridDistribution
from .custom_hyperspherical_distribution import CustomHypersphericalDistribution
from .custom_hyperhemispherical_distribution import CustomHyperhemisphericalDistribution
from .hyperspherical_dirac_distribution import HypersphericalDiracDistribution
from .spherical_harmonics_distribution_complex import (
    SphericalHarmonicsDistributionComplex,
)
from ...sampling.hyperspherical_sampler import LeopardiSampler

import numpy as np
import warnings


class HyperhemisphericalGridDistribution(
    AbstractHypersphereSubsetGridDistribution, AbstractHyperhemisphericalDistribution
):

    def __init__(self, grid_, grid_values_, enforce_pdf_nonnegative=True):
        grid_ = np.asarray(grid_, dtype=float)
        grid_values_ = np.asarray(grid_values_, dtype=float)

        # Close to the MATLAB constructor behaviour
        assert np.all(np.abs(grid_) <= 1 + 1e-12), (
            "Grid points must lie on / inside the unit hypersphere."
        )
        assert np.all(
            grid_[-1, :] >= 0
        ), "Always using upper hemisphere (along last dimension)."

        super().__init__(grid_, grid_values_, enforce_pdf_nonnegative)

    # ------------------------------------------------------------------
    # Basic functionality
    # ------------------------------------------------------------------
    def mean_direction(self):
        """
        For hyperhemispheres this method returns the *mode* (grid point with
        maximum weight) rather than the true mean direction.

        This matches the MATLAB implementation, which warns that using the
        true mean would bias the result toward [0; ...; 0; 1], since the lower
        half of the sphere is not represented.
        """
        warnings.warn(
            "For hyperhemispheres, `mean_direction` returns the mode and "
            "not the true mean direction.",
            UserWarning,
        )
        index_max = np.argmax(self.grid_values)
        mu = self.grid[:, index_max]
        return mu

    def to_full_sphere(self):
        """
        Convert hemisphere to full sphere.

        The grid is mirrored, and the values are halved to keep the resulting
        hyperspherical distribution normalized (just like MATLAB's toFullSphere).
        """
        grid_ = np.hstack((self.grid, -self.grid))
        grid_values_ = 0.5 * np.hstack((self.grid_values, self.grid_values))
        hgd = HypersphericalGridDistribution(grid_, grid_values_)
        return hgd

    # ------------------------------------------------------------------
    # Plotting (same semantics as MATLAB)
    # ------------------------------------------------------------------
    def plot(self):
        hdd = HypersphericalDiracDistribution(self.grid, self.grid_values.T)
        h = hdd.plot()
        return h

    def plot_interpolated(self):
        hdgd = self.to_full_sphere()
        hhgd_interp = CustomHyperhemisphericalDistribution(
            lambda x: 2 * hdgd.pdf(x), 3
        )
        # MATLAB temporarily disables a PDF:UseInterpolated warning here.
        # In Python we just let any interpolation warnings pass through.
        h = hhgd_interp.plot()
        return h

    def plot_full_sphere_interpolated(self):
        if self.dim != 3:
            raise ValueError("Can currently only plot for hemisphere of S2 sphere.")
        hgd = self.to_full_sphere()
        shd = SphericalHarmonicsDistributionComplex.from_grid(
            hgd.grid_values, hgd.grid, "identity"
        )
        chhd = CustomHypersphericalDistribution.from_distribution(shd)
        h = chhd.plot()
        return h

    # ------------------------------------------------------------------
    # Grid geometry utilities
    # ------------------------------------------------------------------
    def get_closest_point(self, xs):
        """
        Return the closest grid point(s) on the hemisphere, taking the symmetry
        x ~ -x into account (like the MATLAB getClosestPoint).

        Parameters
        ----------
        xs : array_like
            Either a single point of shape (dim,),
            or an array of shape (n_points, dim) or (dim, n_points).

        Returns
        -------
        points : ndarray
            Closest grid point(s), shape (dim,) or (dim, n_points).
        indices : ndarray or int
            Indices of the closest grid points.
        """
        xs = np.asarray(xs, dtype=float)

        if xs.ndim == 1:
            if xs.shape[0] != self.dim:
                raise ValueError(
                    f"xs must have length {self.dim}, got {xs.shape[0]}."
                )
            xs = xs[None, :]  # (1, dim)
        elif xs.ndim == 2:
            # Allow both (n_points, dim) and (dim, n_points).
            # We treat (n_points, dim) as the default "batch, dim" layout.
            if xs.shape[1] == self.dim:
                pass  # already (batch, dim)
            elif xs.shape[0] == self.dim:
                xs = xs.T  # (batch, dim)
            else:
                raise ValueError(
                    f"xs must have shape (dim, n) or (n, dim) with dim={self.dim}."
                )
        else:
            raise ValueError("xs must be a 1D or 2D array.")

        # xs: (batch, dim), grid: (dim, n_grid)
        grid_T = self.grid.T  # (n_grid, dim)

        # Distances to each grid point and its antipode.
        diff1 = xs[:, None, :] - grid_T[None, :, :]  # (batch, n_grid, dim)
        diff2 = xs[:, None, :] + grid_T[None, :, :]  # (batch, n_grid, dim)

        dists1 = np.linalg.norm(diff1, axis=2)  # (batch, n_grid)
        dists2 = np.linalg.norm(diff2, axis=2)  # (batch, n_grid)

        all_distances = np.minimum(dists1, dists2)  # (batch, n_grid)

        indices = np.argmin(all_distances, axis=1)  # (batch,)
        points = self.get_grid_point(indices)

        # For a single query, return 1D outputs for convenience.
        if points.ndim == 2 and points.shape[1] == 1:
            points = points[:, 0]
            indices = int(indices[0])

        return points, indices

    # ------------------------------------------------------------------
    # Multiplication on the hemisphere
    # ------------------------------------------------------------------
    def multiply(self, other):
        """
        Multiply two hyperhemispherical grid distributions that share the same grid.

        This mirrors the MATLAB behaviour by:
          1. Converting both to full-sphere grid distributions.
          2. Multiplying them as HypersphericalGridDistribution objects.
          3. Restricting back to the hemisphere and rescaling.

        Parameters
        ----------
        other : HyperhemisphericalGridDistribution

        Returns
        -------
        HyperhemisphericalGridDistribution

        Raises
        ------
        ValueError
            If the grids are not identical (up to numerical tolerance); message
            matches MATLAB's identifier: 'Multiply:IncompatibleGrid'.
        """
        if not isinstance(other, HyperhemisphericalGridDistribution):
            raise TypeError(
                "Can only multiply with another HyperhemisphericalGridDistribution."
            )

        if (
            self.dim != other.dim
            or self.grid.shape != other.grid.shape
            or not np.allclose(self.grid, other.grid)
        ):
            # Mirror MATLAB error identifier
            raise ValueError("Multiply:IncompatibleGrid")

        # 1–2. Multiply on the full sphere using the full-sphere implementation.
        hgd1 = self.to_full_sphere()
        hgd2 = other.to_full_sphere()
        hgd_filtered = hgd1.multiply(hgd2)

        # 3. Restrict to hemisphere and rescale:
        # p_H(x) = 2 * p_S(x) for x on the hemisphere.
        n_hemi = self.grid.shape[1]
        hemi_grid = hgd_filtered.grid[:, :n_hemi]
        hemi_values = 2.0 * hgd_filtered.grid_values[:n_hemi]

        return HyperhemisphericalGridDistribution(
            hemi_grid, hemi_values, enforce_pdf_nonnegative=True
        )

    # ------------------------------------------------------------------
    # Internal helper: "eq_point_set_symm" style grid
    # ------------------------------------------------------------------
    @staticmethod
    def _eq_point_set_symm(dim, n_points, _):
        ls = LeopardiSampler()
        grid, _ = ls.get_grid(n_points * 2, dim)
        return grid

    # ------------------------------------------------------------------
    # Construction from other distributions
    # ------------------------------------------------------------------
    @staticmethod
    def from_distribution(
        distribution, no_of_grid_points, grid_type="eq_point_set_symmetric"
    ):
        from .von_mises_fisher_distribution import VonMisesFisherDistribution
        from .bingham_distribution import BinghamDistribution
        from .hyperspherical_mixture import HypersphericalMixture
        from .watson_distribution import WatsonDistribution

        if isinstance(distribution, AbstractHyperhemisphericalDistribution):
            fun = distribution.pdf
        # pylint: disable=too-many-boolean-expressions
        elif (
            isinstance(distribution, WatsonDistribution)
            or (
                isinstance(distribution, VonMisesFisherDistribution)
                and distribution.mu[-1] == 0
            )
            or isinstance(distribution, BinghamDistribution)
            or (
                isinstance(distribution, HypersphericalMixture)
                and len(distribution.dists) == 2
                and all(w == 0.5 for w in distribution.w)
                and np.array_equal(
                    distribution.dists[1].mu, -distribution.dists[0].mu
                )
            )
        ):
            # Symmetric on the full sphere -> pdf on hemisphere is 2*pdf_full.
            fun = lambda x: 2 * distribution.pdf(x)
        elif isinstance(distribution, HypersphericalGridDistribution):
            raise ValueError(
                "Converting a HypersphericalGridDistribution to a "
                "HyperhemisphericalGridDistribution is not supported"
            )
        elif isinstance(distribution, AbstractHypersphericalDistribution):
            # As in MATLAB: raise a specific warning that can be caught in tests
            warnings.warn(
                "fromDistribution:asymmetricOnHypersphere: "
                "Approximating a hyperspherical distribution on a hemisphere. "
                "The density may not be symmetric. Double check if this is "
                "intentional.",
                UserWarning,
            )
            fun = lambda x: 2 * distribution.pdf(x)
        else:
            raise ValueError("Distribution currently not supported.")

        sgd = HyperhemisphericalGridDistribution.from_function(
            fun, no_of_grid_points, distribution.dim, grid_type
        )
        return sgd

    # ------------------------------------------------------------------
    # Construction from a function handle
    # ------------------------------------------------------------------
    @staticmethod
    def from_function(fun, no_of_grid_points, dim=3, grid_type="eq_point_set_symm"):
        """
        Construct a hyperhemispherical grid distribution from a callable.

        Parameters
        ----------
        fun : callable
            A function mapping an array of shape (batch_dim, space_dim)
            to a 1‑D array of pdf values.  In particular this matches
            your Python convention `pdf(x)` with x.shape == (batch, dim).
        no_of_grid_points : int
            Number of grid points on the hemisphere.
        dim : int, optional
            Ambient dimension (space_dim). Default is 3 (S^2).
        grid_type : {'eq_point_set', 'eq_point_set_symm',
                     'eq_point_set_symmetric', 'healpix'}
            Type of grid to use. The default matches the MATLAB class
            ('eq_point_set_symm').

        Notes
        -----
        For 'eq_point_set*' types, the grid is deterministic and only
        depends on (dim, no_of_grid_points, grid_type). For 'healpix',
        only dim == 3 is supported and `healpy` must be installed.
        """
        if dim < 2:
            raise ValueError("dim must be >= 2")

        # --- eq_point_set-style grids (default) ---
        if grid_type in {"eq_point_set", "eq_point_set_symm", "eq_point_set_symmetric"}:
            # Deterministic pseudo-random grid that depends only on (dim, N, grid_type)
            seed = hash((dim, int(no_of_grid_points), grid_type)) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            grid = HyperhemisphericalGridDistribution._eq_point_set_symm(
                dim, int(no_of_grid_points), rng
            )

        # --- healpix grid (S^2 only, optional) ---
        elif grid_type == "healpix":
            if dim != 3:
                raise ValueError(
                    "healpix grid is only available for dim == 3 (the 2‑sphere)."
                )
            try:
                import healpy as hp  # Imported lazily
            except ImportError as exc:
                raise ImportError(
                    "healpy is required for grid_type='healpix' but is not installed."
                ) from exc

            # Interpret no_of_grid_points as the desired number of hemisphere points.
            target = int(no_of_grid_points)

            # Find smallest nside with at least target pixels on the hemisphere
            nside = 1
            while 6 * nside * nside < target:  # 6 * nside^2 pixels per hemisphere
                nside *= 2

            npix = hp.nside2npix(nside)
            ipix = np.arange(npix, dtype=int)
            theta, phi = hp.pix2ang(nside, ipix)
            vecs = hp.ang2vec(theta, phi).T  # (npix, 3)

            # Keep upper hemisphere (z >= 0)
            hemi_mask = vecs[:, -1] >= 0
            hemi_vecs = vecs[hemi_mask]

            if hemi_vecs.shape[0] < target:
                raise RuntimeError(
                    "Not enough HEALPix pixels on the hemisphere – please choose a larger grid."
                )

            grid = hemi_vecs[:target].T  # (3, target)

        else:
            raise ValueError("Grid scheme not recognized")

        # Respect your Python pdf convention: x has shape (batch_dim, space_dim).
        grid_values = np.asarray(fun(grid.T), dtype=float).reshape(-1)

        sgd = HyperhemisphericalGridDistribution(grid, grid_values)
        return sgd
