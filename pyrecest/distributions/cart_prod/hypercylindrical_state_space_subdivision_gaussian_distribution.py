import copy

# pylint: disable=redefined-builtin,no-name-in-module,no-member
from pyrecest.backend import (
    allclose,
    array,
    atleast_2d,
    concatenate,
)
from pyrecest.backend import sum as backend_sum
from pyrecest.backend import (
    zeros,
)

from ..nonperiodic.gaussian_distribution import GaussianDistribution
from ..nonperiodic.gaussian_mixture import GaussianMixture
from .abstract_hypercylindrical_distribution import AbstractHypercylindricalDistribution


class HypercylindricalStateSpaceSubdivisionGaussianDistribution(
    AbstractHypercylindricalDistribution
):
    """Discrete periodic grid with Gaussian conditionals on the linear part.

    This PR branch does not contain the older
    ``HypercylindricalStateSpaceSubdivisionDistribution`` helper that the
    original draft depended on, nor a generic
    ``StateSpaceSubdivisionGaussianDistribution`` base class. This class
    therefore provides the intended hypertoroidal/linear interface directly in
    a self-contained way.
    """

    def __init__(self, gd_, gaussians):
        if len(gaussians) == 0:
            raise ValueError("gaussians must not be empty")

        AbstractHypercylindricalDistribution.__init__(
            self, bound_dim=gd_.dim, lin_dim=gaussians[0].dim
        )

        if not all(isinstance(g, GaussianDistribution) for g in gaussians):
            raise TypeError(
                "All conditional linear distributions must be GaussianDistribution instances."
            )

        n_grid_points = self._num_grid_points(gd_)
        if n_grid_points != len(gaussians):
            raise ValueError(
                "Number of grid points in gd_ must match number of Gaussian conditionals."
            )

        self.gd = copy.deepcopy(gd_)
        self.gaussians = list(copy.deepcopy(gaussians))
        self.linear_distributions = self.gaussians

    @staticmethod
    def _num_grid_points(gd):
        if hasattr(gd, "d"):
            return gd.d.shape[0]
        if hasattr(gd, "grid_values"):
            return gd.grid_values.shape[0]
        if hasattr(gd, "gridValues"):
            return gd.gridValues.shape[0]
        raise AttributeError("Unable to determine the number of grid points from gd.")

    def _grid_points(self):
        if hasattr(self.gd, "d"):
            grid_points = self.gd.d
        elif hasattr(self.gd, "get_grid"):
            grid_points = self.gd.get_grid()
        elif hasattr(self.gd, "getGrid"):
            grid_points = self.gd.getGrid()
        else:
            raise AttributeError("Unable to extract grid points from gd.")

        if grid_points.ndim == 1:
            grid_points = grid_points.reshape((-1, 1))
        return grid_points

    def _weights(self):
        if hasattr(self.gd, "w"):
            weights = self.gd.w
        elif hasattr(self.gd, "grid_values"):
            weights = self.gd.grid_values
        elif hasattr(self.gd, "gridValues"):
            weights = self.gd.gridValues
        else:
            raise AttributeError("Unable to extract grid weights from gd.")

        return weights / backend_sum(weights)

    def pdf(self, xs):
        xs = atleast_2d(xs)
        if xs.shape[-1] != self.bound_dim + self.lin_dim:
            raise ValueError("xs has the wrong dimension.")

        periodic_inputs = xs[:, : self.bound_dim]
        linear_inputs = xs[:, self.bound_dim :]

        grid_points = self._grid_points()
        weights = self._weights()

        vals = zeros(xs.shape[0])
        for row_idx in range(xs.shape[0]):
            for grid_idx, grid_point in enumerate(grid_points):
                if allclose(periodic_inputs[row_idx], grid_point):
                    vals[row_idx] = weights[grid_idx] * self.gaussians[grid_idx].pdf(
                        linear_inputs[row_idx]
                    )
                    break

        if vals.shape[0] == 1:
            return vals[0]
        return vals

    def marginalize_linear(self):
        return copy.deepcopy(self.gd)

    def marginalize_periodic(self):
        return GaussianMixture(copy.deepcopy(self.gaussians), self._weights())

    def linear_mean(self):
        return self.marginalize_periodic().mean()

    def linear_covariance(self, approximate_mean=None):
        del approximate_mean
        return self.marginalize_periodic().covariance()

    def hybrid_mean(self):
        return concatenate(
            [self.gd.mean_direction().reshape(-1), self.linear_mean().reshape(-1)]
        )

    def hybrid_moment(self):
        trig_mom_complex = self.gd.trigonometric_moment(1)
        trig_mom_real = array([trig_mom_complex.real, trig_mom_complex.imag]).reshape(
            -1
        )
        return concatenate([trig_mom_real, self.linear_mean().reshape(-1)])

    def mode(self):
        periodic_mode = self.gd.mode()
        grid_points = self._grid_points()

        mode_index = None
        for idx, grid_point in enumerate(grid_points):
            if allclose(grid_point.reshape(-1), periodic_mode.reshape(-1)):
                mode_index = idx
                break

        if mode_index is None:
            weights = self._weights()
            mode_index = int(array(weights).argmax())
            periodic_mode = grid_points[mode_index]

        return concatenate(
            [periodic_mode.reshape(-1), self.gaussians[mode_index].mode().reshape(-1)]
        )

    def integrate(self, integration_boundaries=None):
        del integration_boundaries
        return 1.0

    @staticmethod
    def from_distribution(distribution, no_of_grid_points, grid_type="CartesianProd"):
        del distribution, no_of_grid_points, grid_type
        raise NotImplementedError(
            "This PR branch does not contain the non-Gaussian hypercylindrical "
            "state-space subdivision helper required for automatic conversion."
        )

    @staticmethod
    def from_function(  # pylint: disable=too-many-positional-arguments
        fun,
        no_of_grid_points,
        dim_lin,
        dim_bound=1,
        grid_type="CartesianProd",
        int_range=(-float("inf"), float("inf")),
    ):
        del fun, no_of_grid_points, dim_lin, dim_bound, grid_type, int_range
        raise NotImplementedError(
            "This PR branch does not contain the non-Gaussian hypercylindrical "
            "state-space subdivision helper required for construction from a pdf."
        )
