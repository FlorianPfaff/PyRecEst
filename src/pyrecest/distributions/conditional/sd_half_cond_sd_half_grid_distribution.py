import warnings

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    abs,
    all,
    any,
    mean,
    sum,
)
from pyrecest.distributions.hypersphere_subset.abstract_hypersphere_subset_distribution import (
    AbstractHypersphereSubsetDistribution,
)

from .abstract_conditional_distribution import AbstractConditionalDistribution


class SdHalfCondSdHalfGridDistribution(AbstractConditionalDistribution):
    """
    Conditional grid distribution on hemisphere × hemisphere.

    Stores a conditional distribution f(a | b) where both a and b live on
    the upper hemisphere (last coordinate >= 0).

    Convention
    ----------
    - ``grid`` has shape ``(n_points, d)`` where ``d`` is the embedding
      dimension of the individual hemisphere (e.g. d=3 for S2-half).
    - ``grid_values`` has shape ``(n_points, n_points)``.
    - ``grid_values[i, j] = f(grid[i] | grid[j])``.
    - ``dim = 2 * d`` follows the product-space embedding-dimension
      convention from libDirectional.
    """

    def __init__(self, grid, grid_values, enforce_pdf_nonnegative=True):
        """
        Parameters
        ----------
        grid : array of shape (n_points, d)
            Grid points on the upper hemisphere.  All coordinate values must
            lie in ``[-1, 1]`` and the last coordinate must be ``>= 0``.
        grid_values : array of shape (n_points, n_points)
            Conditional pdf values.
        enforce_pdf_nonnegative : bool
            Whether non-negativity of ``grid_values`` is required.
        """
        super().__init__(grid, grid_values, enforce_pdf_nonnegative)
        if any(abs(self.grid) > 1 + 1e-12):
            raise ValueError(
                "Grid points must have coordinates in [-1, 1] (unit sphere)."
            )
        if any(self.grid[:, -1] < -1e-12):
            raise ValueError(
                "Grid points must lie on the upper hemisphere (last coordinate >= 0)."
            )
        self._check_normalization()

    def _check_normalization(self, tol=0.01):
        """Warn if any column is not normalized to 1 over the hemisphere."""
        # Match HyperhemisphericalGridDistribution.get_manifold_size(): a grid
        # in R^d represents S^{d-1}, so the quadrature surface uses d - 1.
        hemisphere_surface = 0.5 * (
            AbstractHypersphereSubsetDistribution.compute_unit_hypersphere_surface(
                self.grid.shape[1] - 1
            )
        )
        ints = mean(self.grid_values, axis=0) * hemisphere_surface
        if any(abs(ints - 1) > tol):
            ints_swapped = mean(self.grid_values, axis=1) * hemisphere_surface
            if all(abs(ints_swapped - 1) <= tol):
                raise ValueError(
                    "Normalization:maybeWrongOrder: Not normalized but would be if "
                    "the order of the two hemispheres were swapped. Check input."
                )
            warnings.warn(
                "Normalization:notNormalized: When conditioning values for the first "
                "hemisphere on the second, normalisation is not ensured. "
                "Check input or increase tolerance.",
                UserWarning,
            )

    def get_grid(self):
        """Return the grid array."""
        return self.grid

    def marginalize_out(self, first_or_second):
        """
        Marginalize out one of the two hemispheres.

        Parameters
        ----------
        first_or_second : int  (1 or 2)

        Returns
        -------
        HyperhemisphericalGridDistribution
        """
        from pyrecest.distributions.hypersphere_subset.hyperhemispherical_grid_distribution import (  # pylint: disable=import-outside-toplevel
            HyperhemisphericalGridDistribution,
        )

        if first_or_second == 1:
            grid_values_hd = sum(self.grid_values, axis=0)
        elif first_or_second == 2:
            grid_values_hd = sum(self.grid_values, axis=1)
        else:
            raise ValueError("first_or_second must be 1 or 2.")

        return HyperhemisphericalGridDistribution(self.grid, grid_values_hd)

    def fix_dim(self, first_or_second, point):
        """
        Return the conditional slice for a fixed grid point.

        Returns
        -------
        HyperhemisphericalGridDistribution
        """
        from pyrecest.distributions.hypersphere_subset.hyperhemispherical_grid_distribution import (  # pylint: disable=import-outside-toplevel
            HyperhemisphericalGridDistribution,
        )

        grid_values_slice = self._get_grid_slice(first_or_second, point)
        return HyperhemisphericalGridDistribution(self.grid, grid_values_slice)

    @staticmethod
    def from_function(
        fun,
        no_of_grid_points,
        fun_does_cartesian_product=False,
        grid_type="leopardi_symm",
        dim=4,
    ):
        """
        Construct a :class:`SdHalfCondSdHalfGridDistribution` from a callable.

        Parameters
        ----------
        fun : callable
            Conditional pdf function ``f(a, b)``.
            When ``fun_does_cartesian_product=False``: called with pairs of
            shape ``(n_pairs, d)`` and must return array of length ``n_pairs``.
            When ``fun_does_cartesian_product=True``: called with both grids
            of shape ``(n_points, d)`` and must return ``(n_points, n_points)``.
        no_of_grid_points : int
            Number of grid points for each hemisphere.
        fun_does_cartesian_product : bool
            See ``fun`` description above.
        grid_type : str
            Grid type for hemisphere. Defaults to ``'leopardi_symm'``.
        dim : int
            Embedding dimension of the Cartesian product space
            (``2 * embedding_dim_of_individual_hemisphere``).
            Default 4 for circles; use 6 for S2-half.

        Returns
        -------
        SdHalfCondSdHalfGridDistribution
        """
        from pyrecest.sampling.hyperspherical_sampler import (  # pylint: disable=import-outside-toplevel
            get_grid_hyperhemisphere,
        )

        n = no_of_grid_points
        manifold_dim = dim // 2 - 1
        grid, _ = get_grid_hyperhemisphere(grid_type, n, manifold_dim)

        grid_values = SdHalfCondSdHalfGridDistribution._evaluate_on_grid(
            fun, grid, n, fun_does_cartesian_product
        )
        return SdHalfCondSdHalfGridDistribution(grid, grid_values)
