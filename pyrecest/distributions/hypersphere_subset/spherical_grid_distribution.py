import numpy as np
import warnings

from .hyperspherical_grid_distribution import HypersphericalGridDistribution
from .abstract_spherical_distribution import AbstractSphericalDistribution
from .spherical_harmonics_distribution_complex import SphericalHarmonicsDistributionComplex
from .custom_hyperspherical_distribution import CustomHypersphericalDistribution
from ..abstract_grid_distribution import AbstractGridDistribution


class SphericalGridDistribution(HypersphericalGridDistribution, AbstractSphericalDistribution):
    """
    Grid-based approximation of a spherical (S²) distribution.

    Conventions:
    - grid: shape (n_points, 3)
    - grid_values: shape (n_points,)
    - pdf(x): x has shape (batch_dim, space_dim) = (N, 3)
    """

    def __init__(self, grid, grid_values, enforce_pdf_nonnegative: bool = True, grid_type: str = "unknown"):
        if self.dim != 2:
            raise AssertionError("SphericalGridDistribution must have dimension 3")

        # Initialize hyperspherical part first
        HypersphericalGridDistribution.__init__(
            self,
            grid,
            grid_values,
            enforce_pdf_nonnegative=enforce_pdf_nonnegative,
            grid_type=grid_type,
        )


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
        return AbstractGridDistribution.normalize(self, tol=tol, warnUnnorm=warn_unnorm)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_interpolated(self, use_harmonics: bool = True):
        """
        Plot interpolated pdf.

        use_harmonics = True  -> spherical harmonics interpolation.
        use_harmonics = False -> piecewise constant interpolation via self.pdf(..., False).
        """
        if use_harmonics:
            if self.enforce_pdf_nonnegative:
                transformation = "sqrt"
            else:
                transformation = "identity"

            shd = SphericalHarmonicsDistributionComplex.from_grid(
                self.grid_values,
                self.grid,
                transformation,
            )
            return shd.plot()
        else:
            chd = CustomHypersphericalDistribution(
                lambda x: self.pdf(x, use_harmonics=False),
                3,
            )
            return chd.plot()

    # ------------------------------------------------------------------
    # Pdf
    # ------------------------------------------------------------------
    def pdf(self, xa, use_harmonics: bool = True):
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
        xa = np.asarray(xa, dtype=float)

        assert xa.shape[0] != self.input_dim

        if use_harmonics:
            warnings.warn(
                "PDF:UseInterpolated: pdf is not defined in closed form for "
                "SphericalGridDistribution; using interpolation with spherical harmonics.",
                UserWarning,
            )

            if self.enforce_pdf_nonnegative:
                transformation = "sqrt"
            else:
                transformation = "identity"

            if self.grid_type != "sh_grid":
                shd = SphericalHarmonicsDistributionComplex.from_grid(
                    self.grid_values,
                    self.grid,
                    transformation,
                )
            else:
                warn_state = warnings.catch_warnings()
                with warn_state:
                    warnings.filterwarnings("ignore", message="Normalization:notNormalized")
                    warnings.warn(
                        "PDF:NeedNormalizationShGrid: Need to normalize for pdf "
                        "because it is unnormalized since sh_grid is used.",
                        UserWarning,
                    )
                    shd = SphericalHarmonicsDistributionComplex.from_grid(
                        self.grid_values,
                        self.grid,
                        transformation,
                    )

            return shd.pdf(xa)

        # use_harmonics == False: piecewise constant interpolation like Hyper/HypersphericalGridDistribution
        warnings.warn(
            "PDF:UseInterpolated: Interpolating the pdf with constant values "
            "in each region is not very efficient, but it is good enough for "
            "visualization purposes.",
            UserWarning,
        )

        dots = self.grid @ xa.T
        max_index = np.argmax(dots, axis=0)
        values = self.grid_values[max_index]
        return float(values[0]) if values.shape[0] == 1 else values

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------
    @staticmethod
    def from_distribution(dist, no_of_grid_points: int, grid_type: str = "eq_point_set",
                          enforce_pdf_nonnegative: bool = True):
        """
        Construct a SphericalGridDistribution from an AbstractHypersphericalDistribution.
        """
        assert dist.dim == 2
        return SphericalGridDistribution.from_function(
            dist.pdf,
            no_of_grid_points,
            grid_type=grid_type,
            enforce_pdf_nonnegative=enforce_pdf_nonnegative,
        )

    @staticmethod
    def from_function(fun, no_of_grid_points: int, grid_type: str = "eq_point_set",
                      enforce_pdf_nonnegative: bool = True):
        """
        Construct from a function fun(x) where x has shape (batch_dim, 3).

        grid_type:
            - 'eq_point_set' : equal point set on S²
            - 'sh_grid'      : spherical harmonics grid (lat/lon mesh)
                               with the same degree formula as MATLAB.
        """
        if grid_type == "eq_point_set":
            # Reuse HypersphericalGridDistribution's generator in 3D
            grid = HypersphericalGridDistribution._eq_point_set(dim=3, n_points=no_of_grid_points,
                                                                rng=np.random.default_rng(0))
        elif grid_type == "sh_grid":
            warnings.warn(
                "Transformation:notEq_Point_set: Not using eq_point_set. "
                "This may lead to problems in the normalization (and filters "
                "based thereon should not be used because the transition may "
                "not be valid).",
                UserWarning,
            )
            # degree = (-6 + sqrt(36 - 8*(4-noOfGridPoints)))/4
            a = -6.0
            b = 36.0 - 8.0 * (4.0 - no_of_grid_points)
            degree = (-a + np.sqrt(b)) / 4.0  # slightly re-arranged but same formula
            if not np.isclose(degree, round(degree)):
                raise ValueError("Number of coefficients not supported for this type of grid.")
            degree = int(round(degree))

            lat = np.linspace(0.0, 2.0 * np.pi, 2 * degree + 2)
            lon = np.linspace(np.pi / 2.0, -np.pi / 2.0, degree + 2)
            lat_mesh, lon_mesh = np.meshgrid(lat, lon)
            # MATLAB: [x,y,z] = sph2cart(latMesh(:)', lonMesh(:)', 1)
            x = np.cos(lon_mesh) * np.cos(lat_mesh)
            y = np.cos(lon_mesh) * np.sin(lat_mesh)
            z = np.sin(lon_mesh)
            grid = np.vstack(
                [x.ravel(), y.ravel(), z.ravel()]
            ).T  # (n_points, 3)
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
