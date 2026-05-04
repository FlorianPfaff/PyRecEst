import builtins
import math
import warnings
from numbers import Integral

from beartype import beartype

# pylint: disable=redefined-builtin
from pyrecest.backend import (
    abs,
    all,
    any,
    argmin,
    array,
    cast,
    int64,
    linspace,
    meshgrid,
    min,
    mod,
    ndim,
    pi,
    reshape,
    stack,
    sum,
)

from ..abstract_grid_distribution import AbstractGridDistribution
from .abstract_hypertoroidal_distribution import AbstractHypertoroidalDistribution
from .hypertoroidal_dirac_distribution import HypertoroidalDiracDistribution


def _normalize_hypertoroidal_resolution(value, dim: int, name: str) -> tuple[int, ...]:
    """Normalize scalar or per-dimension hypertoroidal resolutions."""
    if isinstance(value, Integral):
        resolution = (int(value),) * dim
    else:
        try:
            resolution = tuple(int(v) for v in value)
        except TypeError as exc:
            raise TypeError(
                f"{name} must be an integer or a sequence of integers."
            ) from exc

    if len(resolution) != dim:
        raise ValueError(
            f"{name} must contain one entry per dimension. "
            f"Expected {dim}, got {len(resolution)}."
        )
    if builtins.any(v <= 0 for v in resolution):
        raise ValueError(f"{name} entries must be positive integers.")
    return resolution


class HypertoroidalGridDistribution(
    AbstractGridDistribution, AbstractHypertoroidalDistribution
):
    """Grid-based distribution on a hypertorus.

    Python convention:
        * grid: array of shape (n_grid_points, dim)
        * grid_values: array of shape (n_grid_points,)
        * pdf(x): x has shape (n_eval, dim)

    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        grid_values,
        grid_type: str = "custom",
        grid=None,
        enforce_pdf_nonnegative: bool = True,
        dim: int | None = None,
    ):
        dim = grid_values.ndim

        # Initialize hypertoroidal base class
        AbstractHypertoroidalDistribution.__init__(self, dim)
        # Initialize grid base class
        AbstractGridDistribution.__init__(
            self,
            grid_values,
            grid_type,
            grid,
            dim=dim,
            enforce_pdf_nonnegative=enforce_pdf_nonnegative,
        )

        # Check if normalized. If not: normalize in place.
        self.normalize_in_place()

    # ------------------------------------------------------------------ utils
    def get_closest_point(self, xs):
        """Return the closest grid point (toroidal distance) to x.

        Parameters
        ----------
        x : array_like, shape (dim,) or (1, dim)
        """
        if xs.shape[1] != self.dim:
            raise ValueError(
                f"Expected point of dimension {self.dim}, got {xs.shape[1]}"
            )
        if self.grid is None or self.grid.size == 0:
            raise ValueError("Grid is empty; cannot find closest point.")

        # Toroidal squared distance: min(Δ^2, (2π - |Δ|)^2) summed over dimensions
        delta = self.grid[None, :, :] - xs[:, None, :]  # (1, n_grid, dim)
        abs_delta = abs(delta)
        dists = sum(min((abs_delta**2, (2 * pi - abs_delta) ** 2)), axis=-1)
        min_index = int(argmin(dists[0]))
        return self.grid[min_index]

    def get_manifold_size(self):
        return AbstractHypertoroidalDistribution.get_manifold_size(self)

    # ---------------------------------------------------------- grid helpers
    @staticmethod
    def generate_cartesian_product_grid(n_grid_points):
        """Generate a Cartesian product grid on [0, 2π)^dim.

        Parameters
        ----------
        n_grid_points : int or sequence of int
            Number of grid points along each dimension.
        """
        if any(array(n_grid_points) <= 0):
            raise ValueError("n_grid_points must contain positive integers")

        axes = [linspace(0.0, 2.0 * pi - 2.0 * pi / n, n) for n in n_grid_points]
        mesh = meshgrid(*axes, indexing="ij")
        grid = stack([m.ravel() for m in mesh], axis=-1)  # (n_samples, dim)
        return grid

    # ---------------------------------------------------------------- combine
    def multiply(self, other):
        assert all(
            self.grid == other.grid
        ), "Multiply:IncompatibleGrid: Can only multiply for equal grids."
        return super().multiply(other)

    # pylint: disable=too-many-locals
    def pdf(self, xs):
        """Evaluate the pdf at given query points.

        Parameters
        ----------
        xs : array_like (backend compatible), shape (n_points_eval, dim) or (dim,)
        """
        # 1. Handle shapes using backend functions
        if ndim(xs) == 1:
            xs = reshape(xs, (1, -1))

        if xs.shape[1] != self.dim:
            raise ValueError(
                f"Expected xs with shape (n_points_eval, {self.dim}), got {xs.shape}"
            )

        if self.grid_type != "cartesian_prod":
            return self.value_of_closest(xs)

        # 2. Cartesian product grid logic
        n_grid_points = self.grid_values.shape
        dim = self.dim

        # We collect column indices in a list and stack them at the end.
        indices_cols = []

        for d in range(dim):
            step = 2.0 * pi / n_grid_points[d]

            # Calculate index as float first
            # (val + step/2) // step gives the index, but as a float type
            idx_float = (xs[:, d] + step / 2.0) // step

            # Cast to int64 using the backend's explicit cast function
            idx_int = cast(idx_float, int64)

            # Apply modulo
            idx_wrapped = mod(idx_int, n_grid_points[d])
            indices_cols.append(idx_wrapped)

        # Stack columns to shape (n_points, dim)
        indices = stack(indices_cols, axis=1)

        # 3. Calculate Strides
        # We calculate strides using Python's math.prod on the shape tuple,
        # then convert the result to a backend array.
        strides_list = [
            math.prod(n_grid_points[d + 1 :]) for d in range(dim)  # noqa: E203
        ]
        strides = array(strides_list, dtype=int64)

        # 4. Calculate Flat Indices
        # Sum (indices * strides) along the columns
        flat_indices = sum(indices * strides, axis=1)

        # 5. Index into flattened grid
        # Reshape grid values to 1D
        grid_flat = reshape(self.grid_values, (-1,))

        # Use standard indexing (backends support integer array indexing)
        p = grid_flat[flat_indices]

        return p

    def get_grid(self):
        if self.grid is not None and self.grid.shape[0] > 0:
            return self.grid
        if self.grid_type == "cartesian_prod":
            warnings.warn(
                "Grid:GenerateDuringRunTime: Generating grid anew on call to "
                "get_grid(). If you require the grid frequently, store it in the class.",
                RuntimeWarning,
            )
            if self.n_grid_points is None:
                raise ValueError("Cannot generate grid: n_grid_points is not defined.")
            return self.generate_cartesian_product_grid(self.n_grid_points)
        raise ValueError(
            "Grid:UnknownGrid: Grid was not provided and is thus unavailable"
        )

    def pdf_unnormalized(self, xs):
        if self.grid_type != "cartesian_prod":
            raise ValueError(
                "pdf_unnormalized is only defined for 'cartesian_prod' grids."
            )
        p = self.integrate() * self.pdf(xs)
        return p

    def value_of_closest(self, xa):
        """Return pdf values of the closest grid point for each query.

        Parameters
        ----------
        xa : array_like, shape (n_eval, dim) or (dim,)
        """
        if xa.ndim == 1:
            xa = xa.reshape(1, -1)
        if xa.shape[1] != self.dim:
            raise ValueError(
                f"Expected xa with shape (n_eval, {self.dim}), got {xa.shape}"
            )

        if self.grid is None or self.grid.size == 0:
            raise ValueError("Grid is empty; cannot evaluate value_of_closest.")

        # Broadcast differences: (n_eval, n_grid, dim)
        delta = self.grid[None, :, :] - xa[:, None, :]
        abs_delta = abs(delta)
        dists = sum(min((abs_delta**2, (2 * pi - abs_delta) ** 2)), axis=-1)
        min_inds = argmin(dists, axis=1)
        return self.grid_values[min_inds]

    @staticmethod
    @beartype
    def from_distribution(
        distribution: AbstractHypertoroidalDistribution,
        n_grid_points: int | tuple | list,
        grid_type: str = "cartesian_prod",
        enforce_pdf_nonnegative: bool = True,
    ):
        """Create a grid distribution from an existing hypertoroidal distribution."""
        if not isinstance(distribution, AbstractHypertoroidalDistribution):
            raise ValueError(
                "from_distribution: invalidObject: First argument has to be "
                "a hypertoroidal distribution."
            )
        n_grid_points = _normalize_hypertoroidal_resolution(
            n_grid_points, distribution.dim, "n_grid_points"
        )
        # Generic case: sample pdf of the given distribution on a grid.
        hgd = HypertoroidalGridDistribution.from_function(
            distribution.pdf,
            n_grid_points,
            grid_type=grid_type,
            enforce_pdf_nonnegative=enforce_pdf_nonnegative,
        )
        return hgd

    @staticmethod
    @beartype
    def from_function(
        fun,
        n_grid_points: tuple | list,
        grid_type: str = "cartesian_prod",
        enforce_pdf_nonnegative: bool = True,
    ):
        """Construct a grid distribution by sampling a function on a grid.
        You need to provide the number of grid points along each dimension.
        Dimensionality of the torus is determined by the length of n_grid_points.

        Parameters
        ----------
        fun : callable
            Function handle representing a (possibly unnormalized) pdf.
            Must accept an array of shape (n_eval, dim) and return shape (n_eval,).
        n_grid_points : int or sequence of int
            Number of grid points along each dimension.

        """
        if grid_type == "cartesian_prod":
            grid = HypertoroidalGridDistribution.generate_cartesian_product_grid(
                n_grid_points
            )
        else:
            raise ValueError("Grid scheme not recognized")

        # fun expects points as (prod(n_grid_points), dim) matrix
        grid_values = fun(grid)
        grid_values = reshape(grid_values, n_grid_points)

        sgd = HypertoroidalGridDistribution(
            grid_values=grid_values,
            grid_type=grid_type,
            grid=grid,
            enforce_pdf_nonnegative=enforce_pdf_nonnegative,
        )
        return sgd

    def to_dirac_distribution(self):
        """Return a weighted Dirac distribution on the grid points."""
        weights = reshape(self.grid_values, (-1,))
        weights = weights / sum(weights)
        return HypertoroidalDiracDistribution(self.get_grid(), weights, dim=self.dim)

    def plot(self, *args, **kwargs):
        return self.to_dirac_distribution().plot(*args, **kwargs)

    def trigonometric_moment(self, n):
        hwd = HypertoroidalDiracDistribution(
            self.get_grid(), reshape(self.grid_values, (-1,)) / sum(self.grid_values)
        )
        m = hwd.trigonometric_moment(n)
        return m
