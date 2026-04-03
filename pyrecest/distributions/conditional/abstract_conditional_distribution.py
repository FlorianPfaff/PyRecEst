import copy
import warnings
from abc import ABC

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    any,
    arange,
    argmin,
    array_equal,
    linalg,
    meshgrid,
)


class AbstractConditionalDistribution(ABC):
    """Abstract base class for conditional grid distributions on manifolds.

    Subclasses represent distributions of the form f(a | b) where both a and b
    live on the same manifold.  The joint state is stored as a square matrix
    ``grid_values`` where ``grid_values[i, j] = f(grid[i] | grid[j])``.
    """

    def __init__(self, grid, grid_values, enforce_pdf_nonnegative=True):
        """Common initialisation for conditional grid distributions.

        Parameters
        ----------
        grid : array of shape (n_points, d)
            Grid points on the individual manifold.
        grid_values : array of shape (n_points, n_points)
            Conditional pdf values; ``grid_values[i, j] = f(grid[i] | grid[j])``.
        enforce_pdf_nonnegative : bool
            Whether to require non-negative ``grid_values``.
        """
        if grid.ndim != 2:
            raise ValueError("grid must be a 2D array of shape (n_points, d).")

        n_points, d = grid.shape

        if grid_values.ndim != 2 or grid_values.shape != (n_points, n_points):
            raise ValueError(
                f"grid_values must be a square 2D array of shape ({n_points}, {n_points})."
            )

        if enforce_pdf_nonnegative and any(grid_values < 0):
            raise ValueError("grid_values must be non-negative.")

        self.grid = grid
        self.grid_values = grid_values
        self.enforce_pdf_nonnegative = enforce_pdf_nonnegative
        # Embedding dimension of the Cartesian product space (convention from
        # libDirectional: dim = 2 * dim_of_individual_manifold).
        self.dim = 2 * d

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize(self):
        """No-op – returns ``self`` for compatibility."""
        return self

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def multiply(self, other):
        """Element-wise multiply two conditional grid distributions.

        The resulting distribution is *not* normalized.

        Parameters
        ----------
        other : AbstractConditionalDistribution
            Must be defined on the same grid.

        Returns
        -------
        AbstractConditionalDistribution
            Same concrete type as ``self``.
        """
        if not array_equal(self.grid, other.grid):
            raise ValueError(
                "Multiply:IncompatibleGrid: Can only multiply distributions "
                "defined on identical grids."
            )
        warnings.warn(
            "Multiply:UnnormalizedResult: Multiplication does not yield a "
            "normalized result.",
            UserWarning,
        )
        result = copy.deepcopy(self)
        result.grid_values = result.grid_values * other.grid_values
        return result

    # ------------------------------------------------------------------
    # Protected helpers
    # ------------------------------------------------------------------

    def _get_grid_slice(self, first_or_second, point):
        """Return the ``grid_values`` slice for a fixed grid point.

        Parameters
        ----------
        first_or_second : int  (1 or 2)
            Which variable to fix.
        point : array of shape (d,)
            Must be an existing grid point.

        Returns
        -------
        array of shape (n_points,)
        """
        d = self.grid.shape[1]
        if point.shape[0] != d:
            raise ValueError(f"point must have length {d} (grid dimension).")
        diffs = linalg.norm(self.grid - point[None, :], axis=1)
        locb = argmin(diffs)
        if diffs[locb] > 1e-10:
            raise ValueError(
                "Cannot fix value at this point because it is not on the grid."
            )
        if first_or_second == 1:
            return self.grid_values[locb, :]
        if first_or_second == 2:
            return self.grid_values[:, locb]
        raise ValueError("first_or_second must be 1 or 2.")

    @staticmethod
    def _evaluate_on_grid(fun, grid, n, fun_does_cartesian_product):
        """Evaluate ``fun`` on all grid point pairs and return an (n, n) array.

        Parameters
        ----------
        fun : callable
            ``f(a, b)`` with the semantics described in ``from_function``.
        grid : array of shape (n, d)
            Grid points on the individual manifold.
        n : int
            Number of grid points (``grid.shape[0]``).
        fun_does_cartesian_product : bool
            Whether *fun* handles all grid combinations internally.

        Returns
        -------
        array of shape (n, n)
        """
        if fun_does_cartesian_product:
            fvals = fun(grid, grid)
            return fvals.reshape(n, n)
        idx_a, idx_b = meshgrid(arange(n), arange(n), indexing="ij")
        grid_a = grid[idx_a.ravel()]
        grid_b = grid[idx_b.ravel()]
        fvals = fun(grid_a, grid_b)
        if fvals.shape == (n**2, n**2):
            raise ValueError(
                "Function apparently performs the Cartesian product itself. "
                "Set fun_does_cartesian_product=True."
            )
        return fvals.reshape(n, n)
