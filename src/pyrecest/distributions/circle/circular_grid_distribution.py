import warnings

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    any,
    arange,
    array,
    ceil,
    fft,
    floor,
    isclose,
    linspace,
    maximum,
    mod,
    pi,
    real,
    round,
    sin,
    sqrt,
    sum,
    tile,
    where,
)

from ..abstract_grid_distribution import AbstractGridDistribution
from .abstract_circular_distribution import AbstractCircularDistribution
from .circular_fourier_distribution import CircularFourierDistribution


class CircularGridDistribution(AbstractCircularDistribution, AbstractGridDistribution):
    """
    Density representation using function values on a grid with Fourier interpolation.
    """

    def __init__(self, grid_values, enforce_pdf_nonnegative=True):
        grid_values = array(grid_values)
        if isinstance(grid_values, AbstractCircularDistribution):
            raise ValueError(
                "You gave a distribution as the first argument. "
                "To convert distributions to a distribution in grid representation, "
                "use from_distribution."
            )
        n = grid_values.shape[0]
        grid = linspace(0.0, 2.0 * pi, n, endpoint=False)
        AbstractCircularDistribution.__init__(self)
        AbstractGridDistribution.__init__(
            self,
            grid_values=grid_values,
            grid_type="custom",
            grid=grid,
            dim=1,
            enforce_pdf_nonnegative=enforce_pdf_nonnegative,
        )

    @staticmethod
    def _matlab_sinc(x):
        return where(isclose(x, 0.0), 1.0, sin(x) / x)

    def _pdf_via_sinc(self, xs, sinc_repetitions):
        if sinc_repetitions % 2 != 1:
            raise ValueError("sinc_repetitions must be an odd integer.")

        grid_size = self.grid_values.shape[0]
        step_size = 2.0 * pi / grid_size
        lower = int(floor(sinc_repetitions / 2) * grid_size)
        upper = int(ceil(sinc_repetitions / 2) * grid_size)
        repetitions = arange(-lower, upper)
        sinc_vals = self._matlab_sinc((xs / step_size)[:, None] - repetitions[None, :])
        grid_values = (
            sqrt(self.grid_values) if self.enforce_pdf_nonnegative else self.grid_values
        )
        density = sum(tile(grid_values, sinc_repetitions) * sinc_vals, axis=1)
        if self.enforce_pdf_nonnegative:
            return density**2
        return density

    def _pdf_via_fourier(self, xs):
        transformation = "sqrt" if self.enforce_pdf_nonnegative else "identity"
        function_values = (
            sqrt(self.grid_values) if self.enforce_pdf_nonnegative else self.grid_values
        )
        fd = CircularFourierDistribution.from_function_values(
            function_values, transformation
        )
        return fd.pdf(xs)

    def get_manifold_size(self):
        return 2 * pi

    def get_closest_point(self, xs):
        xs = array(xs)
        n = self.grid_values.shape[0]
        indices = mod(round(xs / (2.0 * pi / n)), n).astype(int)
        points = indices * (2.0 * pi / n)
        return points, indices

    def pdf(self, xs, use_sinc=False, sinc_repetitions=5):
        xs = array(xs)
        if use_sinc:
            return self._pdf_via_sinc(xs, sinc_repetitions)
        return self._pdf_via_fourier(xs)

    @staticmethod
    def from_distribution(distribution, no_of_gridpoints, enforce_pdf_nonnegative=True):
        if isinstance(distribution, CircularFourierDistribution):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fd_to_conv = distribution.truncate(no_of_gridpoints)
            c_shifted = fft.ifftshift(fd_to_conv.c)
            vals_on_grid = real(fft.ifftn(c_shifted)) * (
                len(fd_to_conv.a) + len(fd_to_conv.b)
            )
            if fd_to_conv.transformation == "identity":
                if any(vals_on_grid < 0):
                    warnings.warn("Negative values occurred. Increasing them to 0.")
                    vals_on_grid = maximum(vals_on_grid, 0)
            elif fd_to_conv.transformation == "sqrt":
                vals_on_grid = vals_on_grid**2
            else:
                raise ValueError("Transformation unsupported")
            return CircularGridDistribution(vals_on_grid, enforce_pdf_nonnegative)
        return CircularGridDistribution.from_function(
            distribution.pdf,
            no_of_gridpoints,
            enforce_pdf_nonnegative,
        )

    @staticmethod
    def from_function(fun, no_of_gridpoints, enforce_pdf_nonnegative=True):
        grid_points = linspace(0.0, 2.0 * pi, no_of_gridpoints, endpoint=False)
        grid_values = array(fun(grid_points))
        return CircularGridDistribution(grid_values, enforce_pdf_nonnegative)
