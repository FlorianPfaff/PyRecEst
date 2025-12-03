from collections.abc import Callable
from typing import Union

import matplotlib.pyplot as plt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    array,
    concatenate,
    cos,
    gammaln,
    int32,
    int64,
    linspace,
    log,
    meshgrid,
    ones,
    pi,
    random,
    sin,
    vstack,
    zeros,
    sqrt,
    stack,
    linalg,
)
from scipy.optimize import minimize

from .abstract_hypersphere_subset_distribution import (
    AbstractHypersphereSubsetDistribution,
)


class AbstractHypersphericalDistribution(AbstractHypersphereSubsetDistribution):
    """
    This is an abstract class for a distribution over a hypersphere.
    """

    def mean(self):
        """
        Convenient access to mean_direction to have a consistent interface
        throughout manifolds.

        :return: The mean of the distribution.
        :rtype:
        """
        return self.mean_direction()

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
        """
        Sample from the distribution using Metropolis-Hastings algorithm.

        Args:
            n (int): Number of samples.
            burn_in (int, optional): Number of samples to discard at the start. Defaults to 10.
            skipping (int, optional): Number of samples to skip between each kept sample. Defaults to 5.
            proposal (function, optional): Proposal distribution for the Metropolis-Hastings algorithm. Defaults to None.
            start_point (, optional): Starting point for the Metropolis-Hastings algorithm. Defaults to None.

        Returns:
            : Sampled points.
        """
        if proposal is None:
            # For unimodal densities, other proposals may be far better.
            from .hyperspherical_uniform_distribution import (
                HypersphericalUniformDistribution,
            )

            if pyrecest.backend.__backend_name__ in ("numpy", "pytorch"):
                def proposal(_):
                    return HypersphericalUniformDistribution(self.dim).sample(1)
            else:
                import jax as _jax
                import jax.numpy as _jnp
                def proposal(key, _):
                    """JAX independence proposal: uniform on hypersphere."""
                    if self.dim == 2:
                        # Explicit S² sampling
                        key, key_phi = _jax.random.split(key)
                        key, key_sz = _jax.random.split(key)

                        phi = 2.0 * _jnp.pi * _jax.random.uniform(key_phi, shape=(1,))
                        sz = 2.0 * _jax.random.uniform(key_sz, shape=(1,)) - 1.0
                        r = _jnp.sqrt(1.0 - sz**2)

                        # Shape (1, 3)
                        s = _jnp.stack(
                            [r * _jnp.cos(phi), r * _jnp.sin(phi), sz],
                            axis=1,
                        )
                    else:
                        # General S^d: sample N(0, I) in R^{d+1} and normalize
                        key, subkey = _jax.random.split(key)
                        samples_unnorm = _jax.random.normal(subkey, shape=(1, self.dim + 1))
                        norms = _jnp.linalg.norm(samples_unnorm, axis=1, keepdims=True)
                        s = samples_unnorm / norms

                    return s
                
        if start_point is None:
            start_point = HypersphericalUniformDistribution(self.dim).sample(1)
        # Call the sample_metropolis_hastings method of AbstractDistribution
        # pylint: disable=duplicate-code
        return super().sample_metropolis_hastings(
            n,
            burn_in=burn_in,
            skipping=skipping,
            proposal=proposal,
            start_point=start_point,
        )

    def plot(
        self,
        faces: Union[int, int32, int64] = 100,
        grid_faces: Union[int, int32, int64] = 20,
    ) -> None:
        if self.dim == 1:
            phi = linspace(0, 2 * pi, 320)
            x = array([sin(phi), cos(phi)])
            p = self.pdf(x)
            plt.plot(phi, p)
            plt.show()

        elif self.dim == 2:
            x_sphere_outer, y_sphere_outer, z_sphere_outer = self.create_sphere(
                grid_faces
            )
            x_sphere_inner, y_sphere_inner, z_sphere_inner = self.create_sphere(faces)

            c_sphere = self.pdf(
                array(
                    [
                        x_sphere_inner.flatten(),
                        y_sphere_inner.flatten(),
                        z_sphere_inner.flatten(),
                    ]
                ).T
            ).reshape(x_sphere_inner.shape)

            x_sphere_inner *= 0.99
            y_sphere_inner *= 0.99
            z_sphere_inner *= 0.99

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            if grid_faces > 0:
                print("x_sphere_outer.shape:", x_sphere_outer.shape)
                print("y_sphere_outer.shape:", y_sphere_outer.shape)
                print("z_sphere_outer.shape:", z_sphere_outer.shape)
                ax.plot_surface(
                    x_sphere_outer,
                    y_sphere_outer,
                    z_sphere_outer,
                    facecolors=None,
                    edgecolors="k",
                    alpha=0.1,
                    linewidth=0.5,
                )
            print("x_sphere_inner.shape:", x_sphere_inner.shape)
            print("y_sphere_inner.shape:", y_sphere_inner.shape)
            print("z_sphere_inner.shape:", z_sphere_inner.shape)
            print("c_sphere.shape:", c_sphere.shape)
            ax.plot_surface(
                x_sphere_inner,
                y_sphere_inner,
                z_sphere_inner,
                cmap="viridis",
                rstride=1,
                cstride=1,
                facecolors=plt.cm.viridis(c_sphere),  # pylint: disable=no-member
                shade=False,
            )
            ax.set_box_aspect([1, 1, 1])
            plt.colorbar(plt.cm.ScalarMappable(cmap="viridis"), ax=ax)
            plt.show()
        else:
            raise ValueError(
                "Cannot plot hyperspherical distribution with this number of dimensions."
            )

    def moment(self):
        return self.moment_numerical()

    @staticmethod
    def get_full_integration_boundaries(dim):
        lower_bounds = zeros((dim,))
        upper_bounds = concatenate((array([2 * pi]), pi * ones(dim - 1)))

        return vstack(
            (
                lower_bounds,
                upper_bounds,
            )
        ).T

    def integrate(self, integration_boundaries=None):
        if integration_boundaries is None:
            integration_boundaries = self.__class__.get_full_integration_boundaries(
                self.dim
            )

        return super().integrate(integration_boundaries)

    def integrate_numerically(self, integration_boundaries=None):
        if integration_boundaries is None:
            integration_boundaries = self.__class__.get_full_integration_boundaries(
                self.dim
            )

        return super().integrate_numerically(integration_boundaries)

    def entropy(self):
        return super().entropy_numerical()

    def mode_numerical(self):
        assert (
            pyrecest.backend.__backend_name__ != "jax"
        ), "Not supported on this backend"

        def fun(s):
            return -self.pdf(
                AbstractHypersphereSubsetDistribution.hypersph_to_cart(array(s))
            )

        s0 = random.uniform(size=self.dim) * pi
        res = minimize(
            fun,
            s0,
            method="BFGS",
            options={"disp": False, "gtol": 1e-12, "maxiter": 2000},
        )
        m = AbstractHypersphereSubsetDistribution.hypersph_to_cart(array(res.x))
        return m

    def hellinger_distance(self, other):
        return super().hellinger_distance_numerical(other)

    def total_variation_distance(self, other: "AbstractHypersphericalDistribution"):
        return super().total_variation_distance_numerical(other)

    @staticmethod
    def create_sphere(faces):
        phi_linspace = linspace(0.0, pi, faces)
        theta_linspace = linspace(0.0, 2.0 * pi, faces)
        phi, theta = meshgrid(phi_linspace, theta_linspace, indexing="ij")
        x = sin(phi) * cos(theta)
        y = sin(phi) * sin(theta)
        z = cos(phi)
        return x, y, z

    @staticmethod
    def integrate_fun_over_domain(f_hypersph_coords, dim):
        integration_boundaries = (
            AbstractHypersphericalDistribution.get_full_integration_boundaries(dim)
        )
        return AbstractHypersphereSubsetDistribution.integrate_fun_over_domain_part(
            f_hypersph_coords, integration_boundaries
        )

    def get_manifold_size(self):
        return AbstractHypersphereSubsetDistribution.compute_unit_hypersphere_surface(
            self.dim
        )

    def get_ln_manifold_size(self):
        if self.dim <= 3:
            log_surface_area = log(self.get_manifold_size())
        else:
            # Compute the log of the surface area directly using properties of log
            log_surface_area = (
                log(2.0) + ((self.dim + 1) / 2) * log(pi) - gammaln((self.dim + 1) / 2)
            )
        return log_surface_area
