import warnings

from .hyperspherical_grid_distribution import HypersphericalGridDistribution
from .abstract_spherical_distribution import AbstractSphericalDistribution
from .custom_hyperspherical_distribution import CustomHypersphericalDistribution
from ..abstract_grid_distribution import AbstractGridDistribution

from ...sampling.hyperspherical_sampler import LeopardiSampler, get_grid_hypersphere
from pyrecest.backend import (
    argmax, sqrt
)

class SphericalGridDistribution(HypersphericalGridDistribution, AbstractSphericalDistribution):
    """
    Grid-based approximation of a spherical (S²) distribution.

    Conventions:
    - grid: shape (n_points, 3)
    - grid_values: shape (n_points,)
    - pdf(x): x has shape (batch_dim, space_dim) = (N, 3)
    """
    def __init__(self, grid, grid_values, enforce_pdf_nonnegative: bool = True, grid_type: str = "unknown"):
        AbstractSphericalDistribution.__init__(self)
        super().__init__(grid, grid_values, enforce_pdf_nonnegative=enforce_pdf_nonnegative, grid_type=grid_type)
        if self.dim != 3:
            raise AssertionError("SphericalGridDistribution must have dimension 3")

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    def normalize(self, tol: float = 1e-2, warn_unnorm: bool = True):
        """
        Normalize the grid-based pdf.

        If grid_type == 'sh_grid', we still normalize but warn because the grid
        may not be into equally-sized areas.
        """

        if self.grid_type == "sh_grid":
            warnings.warn(
                "SphericalGridDistribution:CannotNormalizeShGrid: "
                "Cannot properly normalize for sh_grid; using generic normalization anyway."
            )
        return AbstractGridDistribution.normalize(self, tol=tol, warn_unnorm=warn_unnorm)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_interpolated(self, use_harmonics: bool = True):
        """
        Plot interpolated pdf.

        use_harmonics = False -> piecewise constant interpolation via self.pdf(..., False).
        use_harmonics = True  -> spherical harmonics interpolation (currently unsupported).
        """
        assert not use_harmonics, "Using spherical harmonics currently unsupported"
        chd = CustomHypersphericalDistribution(
            lambda x: self.pdf(x, use_harmonics=False),
            3,
        )
        return chd.plot()

    # ------------------------------------------------------------------
    # Pdf
    # ------------------------------------------------------------------
    def pdf(self, xs, use_harmonics: bool = False):
        """
        Pdf on S².

        Parameters
        ----------
        xa : array_like
            (batch_dim, 3).
        use_harmonics : bool
            If True: interpolate via spherical harmonics (preferred).
            If False: piecewise constant interpolation on the grid.
        """
        assert xs.shape[0] != self.input_dim
        assert not use_harmonics, "Using spherical harmonics currently unsupported"

        dots = self.grid @ xs.T
        max_index = argmax(dots, axis=0)
        values = self.grid_values[max_index]
        return float(values[0]) if values.shape[0] == 1 else values

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------
    @staticmethod
    def from_distribution(distribution, no_of_grid_points: int, grid_type: str = "leopardi",
                          enforce_pdf_nonnegative: bool = True):
        """
        Construct a SphericalGridDistribution from an AbstractHypersphericalDistribution.
        """
        assert distribution.dim == 2
        return SphericalGridDistribution.from_function(
            distribution.pdf,
            no_of_grid_points,
            grid_type=grid_type,
            enforce_pdf_nonnegative=enforce_pdf_nonnegative,
        )

    # pylint: disable=too-many-locals
    @staticmethod
    def from_function(fun, no_of_grid_points: int, dim = 2, grid_type: str = "leopardi",
                      enforce_pdf_nonnegative: bool = True):
        """
        Construct from a function fun(x) where x has shape (batch_dim, 3).

        grid_type:
            - 'leopardi' : Leopardi equal point set on S²
            - 'sh_grid'      : spherical harmonics grid (lat/lon mesh).
        """
        assert dim == 2, 'Use HypersphericalGridDistribution for dimensions other than 2.'
        if grid_type == "leopardi":
            # Reuse HypersphericalGridDistribution's generator in 3D
            ls = LeopardiSampler()
            grid, _ = ls.get_grid(no_of_grid_points, 2)
        elif grid_type == "driscoll_healy":
            warnings.warn(
                "Transformation:notLeopardi: Not using leopardi. "
                "This may lead to problems in the normalization (and filters "
                "based thereon should not be used because the transition may "
                "not be valid).",
                UserWarning,
            )
            a = -6.0
            b = 36.0 - 8.0 * (4.0 - no_of_grid_points)
            degree = (-a + sqrt(b)) / 4.0
            grid = get_grid_hypersphere('driscoll_healy', grid_density_parameter=degree, dim=2)
        else:
            raise ValueError("Grid scheme not recognized")

        # fun expects (batch, 3)
        grid_values = fun(grid)
        return SphericalGridDistribution(
            grid,
            grid_values,
            enforce_pdf_nonnegative=enforce_pdf_nonnegative,
            grid_type=grid_type,
        )
