import warnings
from collections.abc import Callable
from typing import Union

import matplotlib.pyplot as plt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    array,
    atleast_1d,
    column_stack,
    diag,
    empty,
    full,
    int32,
    int64,
    linspace,
    meshgrid,
    ndim,
    ones,
    random,
    reshape,
    sqrt,
    squeeze,
    stack,
    zeros,
)
from pyrecest.utils.plotting import plot_ellipsoid
from scipy.integrate import dblquad, nquad, quad
from scipy.optimize import minimize
from scipy.stats import chi2

from ..abstract_manifold_specific_distribution import (
    AbstractManifoldSpecificDistribution,
)


class AbstractLinearDistribution(AbstractManifoldSpecificDistribution):
    def __init__(self, dim: int):
        AbstractManifoldSpecificDistribution.__init__(self, dim)
        self._mean_numerical = None
        self._covariance_numerical = None

    @property
    def input_dim(self):
        return self.dim

    def mean(self):
        if self._mean_numerical is None:
            self._mean_numerical = self.mean_numerical()
        return self._mean_numerical

    def covariance(self):
        if self._covariance_numerical is None:
            self._covariance_numerical = self.covariance_numerical()
        return self._covariance_numerical

    def get_manifold_size(self):
        return float("inf")

    def mode(self, starting_point=None):
        return self.mode_numerical(starting_point)

    def mode_numerical(self, starting_point=None):
        from .gaussian_distribution import GaussianDistribution
        from .gaussian_mixture import GaussianMixture

        assert (
            pyrecest.backend.__backend_name__ == "numpy"
        ), "Only supported for numpy backend"
        if starting_point is None:
            # Take sample if distribution is easy to sample from
            if isinstance(self, (GaussianDistribution, GaussianMixture)):
                # Ensure 1-D for minimize
                starting_point = self.sample(1).squeeze()
            else:
                starting_point = zeros(self.dim)

        def neg_pdf(x):
            return -self.pdf(x)

        assert ndim(starting_point) <= 1, "Starting point must be a 1D array"
        starting_point = atleast_1d(
            starting_point
        )  # Avoid numpy warning "DeprecationWarning: Use of `minimize` with `x0.ndim != 1` is deprecated"

        result = minimize(neg_pdf, starting_point, method="L-BFGS-B")
        return result.x

    # pylint: disable=too-many-positional-arguments
    def sample_metropolis_hastings(
        self,
        n: Union[int, int32, int64],
        burn_in: Union[int, int32, int64] = 10,
        skipping: Union[int, int32, int64] = 5,
        proposal: Callable | None = None,
        start_point=None,
    ):
        if start_point is None:
            if "mean" not in vars(self.__class__) and self._mean_numerical is None:
                # Warn if we need to determine the mean numerically
                warnings.warn(
                    "Starting point for sampling not specified, need to determine the mean numerically."
                )
            start_point = self.mean()

        start_point = atleast_1d(start_point)
        assert start_point.shape == (
            self.input_dim,
        ), "Starting point must be a 1D array of correct dimension"

        if proposal is None:

            proposal_covariance = diag(ones(self.dim))

            if pyrecest.backend.__backend_name__ == "jax":
                from jax import random as jax_random  # pylint: disable=import-error

                def proposal(key, x):
                    noise = jax_random.multivariate_normal(
                        key, zeros(self.dim), proposal_covariance, shape=()
                    )
                    return x + reshape(noise, x.shape)

            else:

                def proposal(x):
                    noise = random.multivariate_normal(
                        zeros(self.dim), proposal_covariance, size=()
                    )
                    return x + reshape(noise, x.shape)

        # pylint: disable=duplicate-code
        return AbstractManifoldSpecificDistribution.sample_metropolis_hastings(
            self,
            n,
            burn_in=burn_in,
            skipping=skipping,
            proposal=proposal,
            start_point=start_point,
        )

    def mean_numerical(self):
        _quad_opts = {"epsabs": 1.49e-7, "epsrel": 1.49e-7}
        if self.dim == 1:
            mu = array(
                quad(
                    lambda x: x * self.pdf(array(x)),
                    array(-float("inf")),
                    array(float("inf")),
                    **_quad_opts,
                )[0]
            )
        elif self.dim == 2:
            mu = array(
                [
                    dblquad(
                        lambda x, y: x * self.pdf(array([x, y])),
                        -float("inf"),
                        float("inf"),
                        lambda _: -float("inf"),
                        lambda _: float("inf"),
                        **_quad_opts,
                    )[0],
                    dblquad(
                        lambda x, y: y * self.pdf(array([x, y])),
                        -float("inf"),
                        float("inf"),
                        lambda _: -float("inf"),
                        lambda _: float("inf"),
                        **_quad_opts,
                    )[0],
                ]
            )
        elif self.dim == 3:
            int_lim = [
                [-float("inf"), float("inf")],
                [-float("inf"), float("inf")],
                [-float("inf"), float("inf")],
            ]

            def integrand1(x, y, z):
                return x * self.pdf(array([x, y, z]))

            def integrand2(x, y, z):
                return y * self.pdf(array([x, y, z]))

            def integrand3(x, y, z):
                return z * self.pdf(array([x, y, z]))

            mu = array(
                [
                    nquad(
                        integrand1,
                        int_lim,
                        opts=_quad_opts,
                    )[0],
                    nquad(
                        integrand2,
                        int_lim,
                        opts=_quad_opts,
                    )[0],
                    nquad(
                        integrand3,
                        int_lim,
                        opts=_quad_opts,
                    )[0],
                ]
            )
        else:
            raise ValueError(
                "Dimension currently not supported for all types of densities."
            )
        return mu

    def covariance_numerical(self):
        _quad_opts = {"epsabs": 1.49e-7, "epsrel": 1.49e-7}
        mu = self.mean()
        if self.dim == 1:
            C = quad(
                lambda x: (x - mu) ** 2 * self.pdf(x),
                -float("inf"),
                float("inf"),
                **_quad_opts,
            )[0]
        elif self.dim == 2:
            C = empty((2, 2))

            def integrand1(x, y):
                return (x - mu[0]) ** 2 * self.pdf(array([x, y]))

            def integrand2(x, y):
                return (x - mu[0]) * (y - mu[1]) * self.pdf(array([x, y]))

            def integrand3(x, y):
                return (y - mu[1]) ** 2 * self.pdf(array([x, y]))

            C[0, 0] = nquad(
                integrand1,
                [[-float("inf"), float("inf")], [-float("inf"), float("inf")]],
                opts=_quad_opts,
            )[0]
            C[0, 1] = nquad(
                integrand2,
                [[-float("inf"), float("inf")], [-float("inf"), float("inf")]],
                opts=_quad_opts,
            )[0]
            C[1, 0] = C[0, 1]
            C[1, 1] = nquad(
                integrand3,
                [[-float("inf"), float("inf")], [-float("inf"), float("inf")]],
                opts=_quad_opts,
            )[0]
        else:
            raise NotImplementedError(
                "Covariance numerical not supported for this dimension."
            )
        return C

    def integrate(self, left=None, right=None):
        if left is None:
            left = -float("inf") * ones(self.dim)
        if right is None:
            right = float("inf") * ones(self.dim)

        result = self.integrate_numerically(left, right)
        return result

    def integrate_numerically(self, left=None, right=None):
        if left is None:
            left = full((self.dim,), -float("inf"))
        if right is None:
            right = full((self.dim,), float("inf"))
        return AbstractLinearDistribution.integrate_fun_over_domain(
            self.pdf, self.dim, left, right
        )

    @staticmethod
    def integrate_fun_over_domain(f, dim, left, right):
        def f_for_nquad(*args):
            # Avoid DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future.
            return squeeze(f(array(args).reshape(-1, dim)))

        if dim == 1:
            result, _ = quad(f_for_nquad, left, right)
        elif dim == 2:
            result, _ = nquad(f_for_nquad, [(left[0], right[0]), (left[1], right[1])])
        elif dim == 3:
            result, _ = nquad(
                f_for_nquad,
                [(left[0], right[0]), (left[1], right[1]), (left[2], right[2])],
            )
        else:
            raise ValueError("Dimension not supported.")
        return result

    def get_suggested_integration_limits(self, scaling_factor=10):
        """
        Returns suggested limits for integration over the whole density.

        The linear part should be integrated from -Inf to Inf but
        Python's numerical integration does not handle that well.
        When we can obtain the covariance of the linear part easily,
        we integrate from mu-10*sigma to mu+scaling_factor*sigma,
        which contains almost the entire probability mass. The
        circular part is integrated form 0 to 2pi.

        Returns:
            l (numpy.ndarray): lower integration bound (shape: (linD+boundD,))
            r (numpy.ndarray): upper integration bound (shape: (linD+boundD,))
        """
        C = self.covariance()
        m = self.mode()
        left = full((self.dim,), float("NaN"))
        right = full((self.dim,), float("NaN"))

        for i in range(self.dim):  # Change for linear dimensions
            left[i] = m[i] - scaling_factor * sqrt(C[i, i])
            right[i] = m[i] + scaling_factor * sqrt(C[i, i])

        return left, right

    def plot(self, *args, plot_range=None, **kwargs):
        mu = self.mean()
        C = self.covariance()

        if plot_range is None:
            scaling = sqrt(chi2.ppf(0.99, self.dim))
            lower_bound = mu - scaling * sqrt(diag(C))
            upper_bound = mu + scaling * sqrt(diag(C))
            plot_range = stack((lower_bound, upper_bound), axis=-1).flatten()

        if self.dim == 1:
            x = linspace(plot_range[0], plot_range[1], 1000)
            y = self.pdf(x)
            plt.plot(x, y, *args, **kwargs)
            plt.show()
        elif self.dim == 2:
            x = linspace(plot_range[0], plot_range[1], 100)
            y = linspace(plot_range[2], plot_range[3], 100)
            x_grid, y_grid = meshgrid(x, y, indexing="ij")
            z_grid = self.pdf(column_stack((x_grid.ravel(), y_grid.ravel())))

            ax = plt.axes(projection="3d")
            ax.plot_surface(
                x_grid, y_grid, reshape(z_grid, x_grid.shape), *args, **kwargs
            )
            plt.show()
        else:
            raise ValueError("Dimension not supported")

    def plot_state(self, scaling_factor=1, color=(0, 0.4470, 0.7410)):
        if self.dim in (
            2,
            3,
        ):
            covariance = self.covariance()
            mean = self.mean()
            plot_ellipsoid(mean, covariance, scaling_factor, color)

        raise ValueError("Dimension currently not supported for plotting the state.")
