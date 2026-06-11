from abc import abstractmethod
from typing import Union

import matplotlib.pyplot as plt
import pyrecest.backend
import scipy.integrate
import scipy.optimize
from matplotlib import cm

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    allclose,
    any,
    arange,
    array,
    asarray,
    column_stack,
    concatenate,
    cos,
    empty,
    full,
    int32,
    int64,
    isnan,
    linspace,
    meshgrid,
    mod,
    ndim,
    ones,
    pi,
    sin,
    sqrt,
    tile,
    vstack,
    zeros,
)
from scipy.integrate import nquad

from ..hypertorus.custom_hypertoroidal_distribution import (
    CustomHypertoroidalDistribution,
)
from ..nonperiodic.custom_linear_distribution import CustomLinearDistribution
from .abstract_lin_periodic_cart_prod_distribution import (
    AbstractLinPeriodicCartProdDistribution,
)


class AbstractHypercylindricalDistribution(AbstractLinPeriodicCartProdDistribution):
    def __init__(
        self, bound_dim: Union[int, int32, int64], lin_dim: Union[int, int32, int64]
    ):
        AbstractLinPeriodicCartProdDistribution.__init__(self, bound_dim, lin_dim)

    @abstractmethod
    def pdf(self, xs):
        pass

    def integrate(self, integration_boundaries=None):
        return self.integrate_numerically(integration_boundaries)

    def integrate_numerically(self, integration_boundaries=None):
        if integration_boundaries is None:
            integration_boundaries = self.get_reasonable_integration_boundaries()

        def f(*args):
            return self.pdf(array(args))

        integration_result = nquad(f, integration_boundaries)[0]

        return integration_result

    def get_reasonable_integration_boundaries(self, scalingFactor=10):
        """
        Return reasonable integration boundaries for numerical integration.

        The input dimensions are ordered as bounded dimensions first and linear
        dimensions second.  SciPy's nquad expects one [lower, upper] interval per
        input dimension, so the periodic dimensions are integrated over a full
        period and the linear dimensions are truncated around the mode according
        to their marginal standard deviations.
        """
        P = self.linear_covariance()
        m = self.mode()

        integration_boundaries = [[0.0, float(2.0 * pi)] for _ in range(self.bound_dim)]

        for linear_idx in range(self.lin_dim):
            input_idx = self.bound_dim + linear_idx
            linear_std = sqrt(P[linear_idx, linear_idx])
            integration_boundaries.append(
                [
                    float(m[input_idx] - scalingFactor * linear_std),
                    float(m[input_idx] + scalingFactor * linear_std),
                ]
            )

        return integration_boundaries

    def mode(self):
        """Find the mode of the distribution by calling mode_numerical."""
        return self.mode_numerical()

    def linear_covariance(self, approximate_mean=None):
        """
        Calculates the linear covariance, or calls linear_covariance_numerical
        if a non-numerical solution doesn't exist.

        Parameters:
        - approximate_mean : ndarray, optional
          The approximate mean to be used. If None, uses NaNs to flag for calculation.

        Returns:
        - C : ndarray
          The linear covariance.
        """
        if approximate_mean is None:
            approximate_mean = full((self.lin_dim,), float("NaN"))
        else:
            approximate_mean = asarray(approximate_mean)

        if approximate_mean.ndim != 1 or approximate_mean.shape[0] != self.lin_dim:
            raise ValueError("approximate_mean must have shape (lin_dim,).")

        return self.linear_covariance_numerical(approximate_mean)

    def linear_covariance_numerical(self, approximate_mean=None):
        """
        Numerically calculates the linear covariance.

        Parameters:
        - approximate_mean : ndarray, optional
          The approximate mean to be used. If None, calculates the mean.

        Returns:
        - C : ndarray
          The linear covariance.
        """
        if approximate_mean is None or any(isnan(approximate_mean)):
            approximate_mean = self.linear_mean_numerical()

        if self.bound_dim == 1 and self.lin_dim == 1:
            C, _ = nquad(
                lambda x, y: (y - approximate_mean) ** 2 * self.pdf([x, y]),
                [[0.0, 2.0 * pi], [-float("inf"), float("inf")]],
            )
        elif self.bound_dim == 2 and self.lin_dim == 1:
            C, _ = nquad(
                lambda x, y, z: (z - approximate_mean) ** 2 * self.pdf([x, y, z]),
                [[0.0, 2.0 * pi], [0.0, 2.0 * pi], [-float("inf"), float("inf")]],
            )
        elif self.bound_dim == 1 and self.lin_dim == 2:
            range_list = [
                [0.0, 2.0 * pi],
                [-float("inf"), float("inf")],
                [-float("inf"), float("inf")],
            ]
            C = empty((2, 2))
            C[0, 0], _ = nquad(
                lambda x, y, z: (y - approximate_mean[0]) ** 2 * self.pdf([x, y, z]),
                range_list,
            )
            C[0, 1], _ = nquad(
                lambda x, y, z: (y - approximate_mean[0])
                * (z - approximate_mean[1])
                * self.pdf([x, y, z]),
                range_list,
            )
            C[1, 0] = C[0, 1]
            C[1, 1], _ = nquad(
                lambda x, y, z: (z - approximate_mean[1]) ** 2 * self.pdf([x, y, z]),
                range_list,
            )
        else:
            raise ValueError("Cannot determine linear covariance for this dimension.")

        return C

    def condition_on_linear(self, input_lin, normalize=True):
        """
        Condition on linear.

        Parameters:
        lin_input : ndarray
            Input array.
        normalize : bool, optional
            If True (default), normalizes the distribution.

        Returns:
        dist : CustomHypertoroidalDistribution
            The distribution after conditioning.
        """
        input_lin = asarray(input_lin)
        if not (
            input_lin.ndim == 0
            and self.lin_dim == 1
            or ndim(input_lin) == 1
            and input_lin.shape[0] == self.lin_dim
        ):
            raise ValueError("Input should be of size (lin_dim,).")

        def f_cond_unnorm(xs, input_lin=input_lin):
            if xs.ndim == 0:
                if self.bound_dim != 1:
                    raise ValueError("Scalar input is only valid for bound_dim == 1.")
                n_inputs = 1
            elif xs.ndim == 1 and self.bound_dim == 1:
                n_inputs = xs.shape[0]
            elif xs.ndim == 1:
                if self.bound_dim != xs.shape[0]:
                    raise ValueError("Input should be of size (bound_dim,).")
                n_inputs = 1
            else:
                n_inputs = xs.shape[0]

            input_repeated = tile(input_lin, (n_inputs, 1))
            return self.pdf(column_stack((xs, input_repeated)))

        dist = CustomHypertoroidalDistribution(f_cond_unnorm, self.bound_dim)

        if normalize:  # Conditional need not be normalized
            dist = dist.normalize()

        return dist

    def condition_on_periodic(self, input_periodic, normalize=True):
        """
        Conditions the distribution on periodic variables

        Arguments:
            input_periodic: ndarray
                Input data, assumed to have shape (self.bound_dim,)
            normalize: bool
                If True, normalizes the distribution

        Returns:
            dist: CustomLinearDistribution
                CustomLinearDistribution instance
        """
        input_periodic = asarray(input_periodic)
        if not (
            input_periodic.ndim == 0
            and self.bound_dim == 1
            or ndim(input_periodic) == 1
            and input_periodic.shape[0] == self.bound_dim
        ):
            raise ValueError(
                "Input should be a scalar for bound_dim == 1 or of size "
                "(bound_dim,)."
            )

        input_periodic = mod(input_periodic, 2.0 * pi)

        def f_cond_unnorm(xs, input_periodic=input_periodic):
            if xs.ndim == 0:
                if self.lin_dim != 1:
                    raise ValueError("Scalar input is only valid for lin_dim == 1.")
                n_inputs = 1
            elif xs.ndim == 1 and self.lin_dim == 1:
                n_inputs = xs.shape[0]
            elif xs.ndim == 1:
                if self.lin_dim != xs.shape[0]:
                    raise ValueError("Input should be of size (lin_dim,).")
                n_inputs = 1
            else:
                n_inputs = xs.shape[0]

            input_repeated = tile(input_periodic, (n_inputs, 1))
            return self.pdf(column_stack((input_repeated, xs)))

        dist = CustomLinearDistribution(f_cond_unnorm, self.lin_dim)

        if normalize:  # Conditional may not be normalized
            dist = dist.normalize()

        return dist

    def linear_mean_numerical(self):
        # Define the integrands for the mean calculation
        if self.lin_dim == 1 and self.bound_dim == 1:
            mu = scipy.integrate.nquad(
                lambda x, y: (y * self.pdf(array([x, y])))[0],
                [[0.0, 2 * pi], [-float("inf"), float("inf")]],
            )[0]
        elif self.bound_dim == 2 and self.lin_dim == 1:
            mu = scipy.integrate.nquad(
                lambda x, y, z: (z * self.pdf([x, y, z]))[0],
                [[0.0, 2 * pi], [0.0, 2 * pi], [-float("inf"), float("inf")]],
            )[0]
        elif self.bound_dim == 1 and self.lin_dim == 2:
            mu = empty(2)
            mu[0] = scipy.integrate.nquad(
                lambda x, y, z: (y * self.pdf([x, y, z]))[0],
                [
                    [0.0, 2 * pi],
                    [-float("inf"), float("inf")],
                    [-float("inf"), float("inf")],
                ],
            )[0]
            mu[1] = scipy.integrate.nquad(
                lambda x, y, z: (z * self.pdf([x, y, z]))[0],
                [
                    [0, 2 * pi],
                    [-float("inf"), float("inf")],
                    [-float("inf"), float("inf")],
                ],
            )[0]
        else:
            raise ValueError("Cannot determine linear mean for this dimension.")

        return mu

    def mode_numerical(self, starting_point=None):
        """
        Find the mode of the distribution numerically.

        Parameters:
        starting_point : ndarray, optional
          The starting point for the optimization.
          If None, uses [pi * ones(self.bound_dim); zeros(self.lin_dim)]

        Returns:
        m : ndarray
          The mode of the distribution.
        """
        if pyrecest.backend.__backend_name__ not in ("numpy", "pytorch"):
            raise NotImplementedError("Not supported for this backend.")
        if starting_point is None:
            starting_point = concatenate(
                [pi * ones(self.bound_dim), zeros(self.lin_dim)]
            )

        # Define bounds for the optimization
        bounds = [
            (0.0, 2.0 * pi) if i < self.bound_dim else (-float("inf"), float("inf"))
            for i in range(self.bound_dim + self.lin_dim)
        ]

        # Perform the optimization
        res = scipy.optimize.minimize(
            lambda x: -self.pdf(array(x)), starting_point, bounds=bounds
        )

        # Check if the optimization might have stopped early
        if allclose(res.x, starting_point):
            print(
                "Warning: Mode was at the starting point. This may indicate the optimizer stopped early."
            )

        return res.x

    @property
    def input_dim(self):
        return self.dim

    def plot(self, *args, **kwargs):
        if self.bound_dim != 1:
            raise NotImplementedError("Plotting is only supported for bound_dim == 1.")

        lin_size = 3
        if self.lin_dim == 1:
            # Creates a three-dimensional surface plot
            step = 2 * pi / 100
            # sigma is the standard deviation of the linear variable
            sigma = sqrt(self.linear_covariance())[0, 0]
            m = self.mode()
            # Create grid over periodic variable (0 to 2*pi)
            x_vals = arange(0, 2 * pi + step, step)
            # Create grid over linear variable (mean +/- lin_size * sigma)
            theta_vals = linspace(
                m[self.bound_dim] - lin_size * sigma,
                m[self.bound_dim] + lin_size * sigma,
                100,
            )
            x, theta = meshgrid(x_vals, theta_vals, indexing="ij")

            # Evaluate pdf at the grid points
            points = vstack((x.ravel(), theta.ravel())).T
            f = self.pdf(points)
            f = f.reshape(x.shape)

            # Now plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(x, theta, f, *args, **kwargs)
            ax.set_xlim([0, 2 * pi])
            ax.set_xlabel("Periodic Variable (Radians)")
            ax.set_ylabel("Linear Variable")
            ax.set_zlabel("PDF Value")
            plt.show()
        elif self.lin_dim == 2:
            raise NotImplementedError("Plotting not supported for lin_dim == 2.")
        else:
            raise NotImplementedError("Plotting not supported for this lin_dim.")

    # pylint: disable=too-many-locals
    def plot_cylinder(self, limits_linear=None):
        if not (self.bound_dim == 1 and self.lin_dim == 1):
            raise NotImplementedError(
                "plot_cylinder is only implemented for bound_dim == 1 and lin_dim == 1."
            )

        if limits_linear is None:
            scale_lin = 3
            m = self.linear_mean()
            P = self.linear_covariance()
            if not isnan(m).any() and not isnan(P).any():
                limits_linear = [
                    m[0] - scale_lin * sqrt(P[0, 0]),
                    m[0] + scale_lin * sqrt(P[0, 0]),
                ]
            else:
                # Sample to find a suitable range
                s = self.sample(100)
                # s is an array of shape (dim, N)
                limits_linear = [
                    s[self.bound_dim :, :].min(),  # noqa: E203
                    s[self.bound_dim :, :].max(),  # noqa: E203
                ]

        phi = linspace(0.0, 2 * pi, 100)
        lin = linspace(limits_linear[0], limits_linear[1], 100)
        Phi, L = meshgrid(phi, lin, indexing="ij")
        points = vstack([Phi.ravel(), L.ravel()]).T
        C = self.pdf(points)
        C = C.reshape(Phi.shape)

        X = cos(Phi)
        Y = sin(Phi)
        Z = L

        # Now plot using surf
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # Normalize C to 0-1
        norm = plt.Normalize(C.min(), C.max())
        # Map normalized values to colors
        colors = cm.viridis(norm(C))

        surf = ax.plot_surface(
            X, Y, Z, facecolors=colors, linewidth=0, antialiased=False, shade=False
        )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Linear Variable")
        plt.show()

        return surf
