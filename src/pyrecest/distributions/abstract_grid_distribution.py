import copy
import warnings
from abc import abstractmethod
from math import prod

from beartype import beartype

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import abs, allclose, any, mean

from .abstract_distribution_type import AbstractDistributionType


class AbstractGridDistribution(AbstractDistributionType):
    # pylint: disable=too-many-positional-arguments
    @beartype
    def __init__(
        self,
        grid_values,
        grid_type: str = "custom",
        grid=None,
        dim=None,
        enforce_pdf_nonnegative: bool = True,
    ):
        if grid_type == "custom" and grid is None:
            raise ValueError("Custom grids require grid coordinates.")
        if grid is not None and grid.shape != ():
            # Use builtin prod because .shape is a tuple of ints
            expected_grid_points = prod(grid_values.shape)
            if grid.shape[0] != expected_grid_points:
                raise ValueError(
                    "The number of grid coordinates must match the number of "
                    f"grid values. Expected {expected_grid_points}, got "
                    f"{grid.shape[0]}."
                )
            if grid.ndim != 1 and grid.shape[1] != dim:
                raise ValueError(
                    f"Grid coordinates must have dimension {dim}, got "
                    f"{grid.shape[1]}."
                )
        if grid is None or grid.ndim > 1 and grid.shape[0] < grid.shape[1]:
            warnings.warn(
                "Warning: Dimension is higher than number of grid points. Verify that this is really intended."
            )
        self.grid_values = grid_values
        self.grid_type = grid_type
        self.grid = grid
        self.enforce_pdf_nonnegative = enforce_pdf_nonnegative
        # Overwrite with more descriptive parameterization
        self.grid_density_description = {
            "n_grid_values": grid_values.shape[0],
            "grid_type": grid_type,
        }

    def pdf(self, xs):
        # Use nearest neighbor interpolation by default
        _, indices = self.get_closest_point(xs)
        return self.grid_values[indices].T

    @property
    def n_grid_points(self):
        # Overwrite if grid_values contains values that are not used as grid values
        return self.grid_values.shape[0]

    @abstractmethod
    def get_closest_point(self, xs):
        pass

    @abstractmethod
    def get_manifold_size(self):
        pass

    def integrate(self, integration_boundaries=None):
        if integration_boundaries is not None:
            raise NotImplementedError(
                "Custom integration boundaries are currently not supported."
            )
        return self.get_manifold_size() * mean(self.grid_values)

    def normalize_in_place(self, tol=1e-4, warn_unnorm=True):
        int_val = self.integrate()
        if float(abs(int_val)) < 1e-200:
            raise ValueError(
                "Sum of grid values is too close to zero, this usually points to a user error."
            )
        if any(self.grid_values < 0):
            warnings.warn(
                "Warning: There are negative values. This usually points to a user error."
            )
        elif abs(int_val - 1) > tol:
            if warn_unnorm:
                warnings.warn(
                    "Warning: Grid values apparently do not belong to a normalized density. Normalizing..."
                )

        self.grid_values = self.grid_values / int_val
        return self

    def normalize(self, tol=1e-4, warn_unnorm=True):
        result = copy.deepcopy(self)
        return result.normalize_in_place(tol=tol, warn_unnorm=warn_unnorm)

    def get_grid(self):
        # Overload if .grid should stay empty
        return self.grid

    def get_grid_point(self, indices):
        # To avoid passing all points if only one or few are needed.
        # Overload if .grid should stay empty
        return self.grid[indices, :]

    def multiply(self, other):
        if not isinstance(other, AbstractGridDistribution):
            raise TypeError("other must be an AbstractGridDistribution.")
        if self.enforce_pdf_nonnegative != other.enforce_pdf_nonnegative:
            raise ValueError(
                "Both grid distributions must agree on enforce_pdf_nonnegative."
            )
        if self.grid_values.shape != other.grid_values.shape:
            raise ValueError("Grid value shapes must match before multiplication.")
        if self.grid_type != other.grid_type:
            raise ValueError("Grid types must match before multiplication.")
        if (self.grid is None) != (other.grid is None):
            raise ValueError(
                "Both grid distributions must either store grids or omit them."
            )
        if self.grid is not None and not allclose(self.grid, other.grid):
            raise ValueError("Grid coordinates must match before multiplication.")
        gd = copy.deepcopy(self)
        gd.grid_values = gd.grid_values * other.grid_values
        gd = gd.normalize(warn_unnorm=False)
        return gd
