from .sd_cond_sd_grid_distribution import SdCondSdGridDistribution


class S2CondS2GridDistribution(SdCondSdGridDistribution):
    """
    Conditional distribution on S2 x S2 represented by a grid of values.

    This is a specialisation of :class:`SdCondSdGridDistribution` for the
    two-sphere (S²).  The grid is restricted to embedding dimension 3
    (``grid.shape[1] == 3``), and factory / slicing methods return
    :class:`~pyrecest.distributions.hypersphere_subset.spherical_grid_distribution.SphericalGridDistribution`
    instances instead of the generic
    :class:`~pyrecest.distributions.hypersphere_subset.hyperspherical_grid_distribution.HypersphericalGridDistribution`.
    """

    def __init__(self, grid, grid_values, enforce_pdf_nonnegative=True):
        """
        Parameters
        ----------
        grid : array of shape (n_points, 3)
            Grid points on S².
        grid_values : array of shape (n_points, n_points)
            Conditional pdf values: ``grid_values[i, j] = f(grid[i] | grid[j])``.
        enforce_pdf_nonnegative : bool
            Whether non-negativity of ``grid_values`` is required.
        """
        if grid.ndim != 2 or grid.shape[1] != 3:
            raise ValueError(
                "S2CondS2GridDistribution requires a grid of shape (n_points, 3)."
            )
        super().__init__(grid, grid_values, enforce_pdf_nonnegative)

    # ------------------------------------------------------------------
    # Marginalisation and conditioning – return SphericalGridDistribution
    # ------------------------------------------------------------------

    def marginalize_out(self, first_or_second):
        """
        Marginalize out one of the two spheres.

        Returns a :class:`SphericalGridDistribution` (S²-specific).

        Parameters
        ----------
        first_or_second : int  (1 or 2)
        """
        # pylint: disable=import-outside-toplevel
        from pyrecest.distributions.hypersphere_subset.spherical_grid_distribution import (
            SphericalGridDistribution,
        )

        hgd = super().marginalize_out(first_or_second)
        return SphericalGridDistribution(hgd.grid, hgd.grid_values)

    def fix_dim(self, first_or_second, point):
        """
        Return the conditional slice for a fixed grid point.

        Returns a :class:`SphericalGridDistribution` (S²-specific).

        Parameters
        ----------
        first_or_second : int  (1 or 2)
        point : array of shape (3,)
        """
        # pylint: disable=import-outside-toplevel
        from pyrecest.distributions.hypersphere_subset.spherical_grid_distribution import (
            SphericalGridDistribution,
        )

        hgd = super().fix_dim(first_or_second, point)
        return SphericalGridDistribution(hgd.grid, hgd.grid_values)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def from_function(
        fun,
        no_of_grid_points,
        fun_does_cartesian_product=False,
        grid_type="leopardi",
        dim=6,
    ):
        """
        Construct an :class:`S2CondS2GridDistribution` from a callable.

        Parameters
        ----------
        fun : callable
            Conditional pdf ``f(a, b)`` – see
            :meth:`SdCondSdGridDistribution.from_function` for the
            ``fun_does_cartesian_product`` convention.
        no_of_grid_points : int
            Number of grid points for each sphere.
        fun_does_cartesian_product : bool
            If ``True``, ``fun`` is called with the full grids of shape
            ``(n_points, 3)`` and must return ``(n_points, n_points)``.
            If ``False`` (default), ``fun`` receives paired rows and must
            return a 1-D array.
        grid_type : str
            Grid type passed to the sampler.  Defaults to ``'leopardi'``.

        Returns
        -------
        S2CondS2GridDistribution
        """
        if dim != 6:
            raise ValueError("S2CondS2GridDistribution is fixed to dim=6.")

        sdsd = SdCondSdGridDistribution.from_function(
            fun,
            no_of_grid_points,
            fun_does_cartesian_product,
            grid_type,
            dim=dim,
        )
        return S2CondS2GridDistribution(sdsd.grid, sdsd.grid_values)
