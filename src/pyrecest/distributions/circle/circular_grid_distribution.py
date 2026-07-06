from numbers import Integral

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    any,
    arange,
    array,
    atleast_1d,
    ceil,
    floor,
    isclose,
    linspace,
    mod,
    ndim,
    pi,
    reshape,
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


def _validate_no_of_gridpoints(no_of_gridpoints) -> int:
    if isinstance(no_of_gridpoints, bool) or not isinstance(no_of_gridpoints, Integral):
        raise ValueError("no_of_gridpoints must be a positive integer.")
    no_of_gridpoints = int(no_of_gridpoints)
    if no_of_gridpoints <= 0:
        raise ValueError("no_of_gridpoints must be a positive integer.")
    return no_of_gridpoints


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
        if enforce_pdf_nonnegative and any(grid_values < 0):
            raise ValueError(
                "grid_values must be nonnegative when "
                "enforce_pdf_nonnegative=True."
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
        zero_mask = isclose(x, 0.0)
        scaled_x = pi * x
        safe_scaled_x = where(zero_mask, 1.0, scaled_x)
        return where(zero_mask, 1.0, sin(scaled_x) / safe_scaled_x)

    def _pdf_via_sinc(self, xs, sinc_repetitions):
        sinc_repetitions = _validate_no_of_gridpoints(sinc_repetitions)
        if sinc_repetitions % 2 != 1:
            raise ValueError("sinc_repetitions must be a positive odd integer.")

        scalar_input = ndim(xs) == 0
        xs_eval = reshape(atleast_1d(xs), (-1,))

        grid_size = self.grid_values.shape[0]
        step_size = 2.0 * pi / grid_size
        lower = int(floor(sinc_repetitions / 2) * grid_size)
        upper = int(ceil(sinc_repetitions / 2) * grid_size)
        repetitions = arange(-lower, upper)
        sinc_vals = self._matlab_sinc(
            (xs_eval / step_size)[:, None] - repetitions[None, :]
        )
        grid_values = (
            sqrt(self.grid_values) if self.enforce_pdf_nonnegative else self.grid_values
        )
        density = sum(tile(grid_values, sinc_repetitions) * sinc_vals, axis=1)
        if self.enforce_pdf_nonnegative:
            density = density**2
        if scalar_input:
            return density[0]
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

    def get_grid_point(self, indices):
        return self.grid[indices]

    def trigonometric_moment(self, n):
        weights = self.grid_values / sum(self.grid_values)
        grid = self.get_grid()
        return sum(weights * (sin(pi / 2 + n * grid) + 1j * sin(n * grid)))

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
        return CircularGridDistribution.from_function(
            distribution.pdf,
            no_of_gridpoints,
            enforce_pdf_nonnegative,
        )

    @staticmethod
    def from_function(fun, no_of_gridpoints, enforce_pdf_nonnegative=True):
        no_of_gridpoints = _validate_no_of_gridpoints(no_of_gridpoints)
        grid_points = linspace(0.0, 2.0 * pi, no_of_gridpoints, endpoint=False)
        grid_values = array(fun(grid_points))
        return CircularGridDistribution(grid_values, enforce_pdf_nonnegative)
