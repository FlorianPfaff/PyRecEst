from collections.abc import Callable
from typing import Union

import matplotlib.pyplot as plt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    abs,
    angle,
    arange,
    array,
    atleast_2d,
    column_stack,
    cos,
    int32,
    int64,
    isnan,
    linspace,
    log,
    meshgrid,
    minimum,
    mod,
    ones,
    pi,
    random,
    sin,
    sqrt,
    vstack,
    zeros,
)
from scipy.integrate import nquad

from ..abstract_manifold_specific_distribution import (
    AbstractManifoldSpecificDistribution,
)
from ..abstract_periodic_distribution import AbstractPeriodicDistribution


class AbstractHypertoroidalDistribution(AbstractPeriodicDistribution):
    """An abstract class representing a Hypertoroidal Distribution"""

    @property
    def input_dim(self) -> int:
        return self.dim

    @staticmethod
    def integrate_fun_over_domain(f: Callable, dim: Union[int, int32, int64]) -> float:
        integration_boundaries = column_stack((zeros(dim), 2 * pi * ones(dim)))
        return AbstractHypertoroidalDistribution.integrate_fun_over_domain_part(
            f, integration_boundaries
        )

    def shift(self, shift_by):
        """
        Shift the distribution by a given vector.

        :param shift_by: The shift vector. Must be of the same dimension as the distribution.
        :type shift_by:

        :return: The shifted distribution.
        :rtype: CustomHypertoroidalDistribution

        :raises AssertionError: If the shift vector is not of the same dimension as the distribution.
        """
        from .custom_hypertoroidal_distribution import CustomHypertoroidalDistribution

        assert shift_by.shape == (
            self.dim,
        ), "Shift vector must be of the same dimension as the distribution."

        # Define the shifted PDF. A positive shift moves mass forward on the
        # hypertorus, hence the shifted density evaluates the original density
        # at the inverse-shifted query point.
        def shifted_pdf(xs):
            return self.pdf(mod(xs - shift_by, 2 * pi))

        # Create the shifted distribution
        shifted_distribution = CustomHypertoroidalDistribution(shifted_pdf, self.dim)

        return shifted_distribution

    @staticmethod
    def integrate_fun_over_domain_part(f: Callable, integration_boundaries) -> float:
        integration_boundaries = atleast_2d(integration_boundaries)
        assert integration_boundaries.shape[-1] == 2

        def scalar_f(*args):
            return f(*args).item()  # ensures 0-dim scalar

        return nquad(scalar_f, integration_boundaries)[0]

    def integrate_numerically(self, integration_boundaries=None):
        assert (
            pyrecest.backend.__backend_name__ == "numpy"
        ), "Only supported for numpy backend"
        if integration_boundaries is None:
            return self.integrate_fun_over_domain(
                lambda *args: self.pdf(array(args)), self.dim
            )

        assert self.dim in (1, integration_boundaries.shape[0])
        return self.integrate_fun_over_domain_part(
            lambda *args: self.pdf(array(args)), integration_boundaries
        )

    def trigonometric_moment_numerical(self, n: Union[int, int32, int64]):
        """Calculates the complex trignometric moments. Since nquad does not support complex functions,
        the calculation is split up (as used in the alternative representation of trigonometric polonymials
        involving the two real numbers alpha and beta"""

        def moment_fun_real(*args):
            x = array(args)
            return array([self.pdf(x) * cos(n * xi) for xi in x])

        def moment_fun_imag(*args):
            x = array(args)
            return array([self.pdf(x) * sin(n * xi) for xi in x])

        alpha = zeros(self.dim, dtype=float)
        beta = zeros(self.dim, dtype=float)

        for i in range(self.dim):
            # i=i to avoid pylint warning (though it does not matter here)
            alpha[i] = self.integrate_fun_over_domain(
                lambda *args, i=i: moment_fun_real(*args)[i], self.dim
            )
            beta[i] = self.integrate_fun_over_domain(
                lambda *args, i=i: moment_fun_imag(*args)[i], self.dim
            )

        return alpha + 1j * beta

    def entropy_numerical(self):
        def entropy_fun(*args):
            x = array(args)
            pdf_val = self.pdf(x)
            return pdf_val * log(pdf_val)

        return -self.integrate_fun_over_domain(entropy_fun, self.dim)

    def get_manifold_size(self):
        return (2.0 * pi) ** self.dim

    @staticmethod
    def angular_error(alpha, beta):
        """
        Calculates the angular error between alpha and beta.

        Parameters:
            alpha (float or numpy array): The first angle(s) in radians.
            beta (float or numpy array): The second angle(s) in radians.

        Returns:
            float or numpy array: The angular error(s) in radians.
        """
        assert not isnan(alpha).any() and not isnan(beta).any()
        # Ensure the angles are between 0 and 2*pi
        alpha = mod(alpha, 2.0 * pi)
        beta = mod(beta, 2.0 * pi)

        # Calculate the absolute difference
        diff = abs(alpha - beta)

        # Calculate the angular error
        e = minimum(diff, 2.0 * pi - diff)

        return e

    def hellinger_distance_numerical(self, other):
        assert isinstance(other, AbstractHypertoroidalDistribution)
        assert (
            self.dim == other.dim
        ), "Cannot compare distributions with different number of dimensions."

        def hellinger_dist_fun(*args):
            x = array(args)
            return (sqrt(self.pdf(x)) - sqrt(other.pdf(x))) ** 2

        squared_dist = 0.5 * self.integrate_fun_over_domain(
            hellinger_dist_fun, self.dim
        )
        return sqrt(squared_dist)

    def total_variation_distance_numerical(self, other):
        assert isinstance(other, AbstractHypertoroidalDistribution)
        assert (
            self.dim == other.dim
        ), "Cannot compare distributions with different number of dimensions"

        def total_variation_dist_fun(*args):
            x = array(args)
            return abs(self.pdf(x) - other.pdf(x))

        dist = 0.5 * self.integrate_fun_over_domain(total_variation_dist_fun, self.dim)
        return dist

    def plot(self, resolution=128, **kwargs):
        if self.dim == 1:
            theta = linspace(0.0, 2 * pi, resolution)
            f_theta = self.pdf(theta)
            p = plt.plot(theta, f_theta, **kwargs)
            AbstractHypertoroidalDistribution.setup_axis_circular("x")
        elif self.dim == 2:
            step = 2 * pi / resolution
            alpha, beta = meshgrid(
                arange(0.0, 2.0 * pi, step), arange(0.0, 2.0 * pi, step), indexing="ij"
            )
            f = self.pdf(column_stack((alpha.ravel(), beta.ravel())))
            f = f.reshape(alpha.shape)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            surf = ax.plot_surface(alpha, beta, f, cmap="viridis", edgecolor="none")
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            AbstractHypertoroidalDistribution.setup_axis_circular("x")
            AbstractHypertoroidalDistribution.setup_axis_circular("y")
            p = ax
        elif self.dim == 3:
            raise NotImplementedError(
                "Plotting for this dimension is currently not supported"
            )
        else:
            raise ValueError("Plotting for this dimension is currently not supported")
        plt.show()
        return p

    def mean(self):
        """
        Convenient access to mean_direction to have a consistent interface
        throughout manifolds.

        :return: The mean of the distribution.
        :rtype:
        """
        return self.mean_direction()

    def mean_direction(self):
        a = self.trigonometric_moment(1)
        m = mod(angle(a), 2.0 * pi)
        return m

    def mode(self):
        return self.mode_numerical()

    def mode_numerical(self):
        # Implement the optimization function fminunc equivalent in Python (e.g., using scipy.optimize.minimize)
        raise NotImplementedError("Mode calculation is not implemented")

    def trigonometric_moment(self, n: Union[int, int32, int64]):
        return self.trigonometric_moment_numerical(n)

    def integrate(self, integration_boundaries=None):
        return self.integrate_numerically(integration_boundaries)

    def mean_2dimD(self):
        m = self.trigonometric_moment_numerical(1)
        mu = vstack((m.real, m.imag))
        return mu

    # jscpd:ignore-start
    # pylint: disable=too-many-positional-arguments
    def sample_metropolis_hastings(
        self,
        n: Union[int, int32, int64],
        burn_in: Union[int, int32, int64] = 10,
        skipping: Union[int, int32, int64] = 5,
        proposal: Callable | None = None,
        start_point=None,
    ):
        # jscpd:ignore-end
        if proposal is None:
            if pyrecest.backend.__backend_name__ == "jax":
                import jax as _jax  # pylint: disable=import-error
                import jax.numpy as _jnp  # pylint: disable=import-error

                def proposal_jax(key, x):
                    key, subkey = _jax.random.split(key)
                    noise = _jax.random.normal(subkey, shape=(self.dim,))
                    return _jnp.mod(x + noise, 2.0 * _jnp.pi)

                proposal = proposal_jax
            else:

                def proposal_np(x):
                    return mod(x + random.normal(0.0, 1.0, (self.dim,)), 2.0 * pi)

                proposal = proposal_np

        if start_point is None:
            start_point = self.mean_direction()

        # pylint: disable=duplicate-code
        s = AbstractManifoldSpecificDistribution.sample_metropolis_hastings(
            self, n, burn_in, skipping, proposal=proposal, start_point=start_point
        )
        return s

    @staticmethod
    def setup_axis_circular(axis_name: str = "x", ax=plt.gca()) -> None:
        ticks = [0.0, pi, 2.0 * pi]
        tick_labels = ["0", r"$\pi$", r"$2\pi$"]
        if axis_name == "x":
            ax.set_xlim(0.0, 2.0 * pi)
            ax.set_xticks(ticks)
            ax.set_xticklabels(tick_labels)
        elif axis_name == "y":
            ax.set_ylim(0.0, 2.0 * pi)
            ax.set_yticks(ticks)
            ax.set_yticklabels(tick_labels)
        elif axis_name == "z":
            ax.set_zlim(0.0, 2.0 * pi)
            ax.set_zticks(ticks)
            ax.set_zticklabels(tick_labels)
        else:
            raise ValueError("invalid axis")
