import copy
from collections.abc import Callable
from operator import index as operator_index

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    argmax,
    array,
    is_bool,
    linalg,
    ndim,
    outer,
    reshape,
    stack,
    sum,
    where,
)

from ..abstract_dirac_distribution import AbstractDiracDistribution
from ..hypersphere_subset.abstract_hypersphere_subset_distribution import (
    AbstractHypersphereSubsetDistribution,
)
from .abstract_cart_prod_distribution import AbstractCartProdDistribution


class HyperhemisphereCartProdDiracDistribution(
    AbstractDiracDistribution, AbstractCartProdDistribution
):
    def __init__(
        self, d, w=None, dim_hemisphere=None, n_hemispheres=None, *, store_flat=True
    ):
        """
        Initialize a Dirac distribution with given Dirac locations and weights.

        :param d: Dirac locations as a numpy array.
        :param w: Weights of Dirac locations as a numpy array. If not provided, defaults to uniform weights.
        """
        if dim_hemisphere is None:
            raise ValueError("Hemisphere dimension must be specified.")
        if n_hemispheres is None:
            raise ValueError("Number of hemispheres must be specified.")
        self.dim_hemisphere = dim_hemisphere
        self.n_hemispheres = n_hemispheres
        self._store_flat = store_flat
        particles = self._as_component_array(d, dim_hemisphere, n_hemispheres)
        if store_flat:
            particles = reshape(
                particles,
                (particles.shape[0], self.input_dim),
            )

        super().__init__(particles, w)
        self.dim = dim_hemisphere * n_hemispheres

    @property
    def component_dim(self):
        return self.dim_hemisphere + 1

    @property
    def input_dim(self):
        return self.component_dim * self.n_hemispheres

    @staticmethod
    def _as_component_array(d, dim_hemisphere, n_hemispheres):
        particles = array(d)
        component_dim = dim_hemisphere + 1
        input_dim = component_dim * n_hemispheres

        if particles.ndim == 2:
            if particles.shape[-1] != input_dim:
                raise ValueError("Dimension is not correct.")
            return reshape(
                particles, (particles.shape[0], n_hemispheres, component_dim)
            )

        if particles.ndim == 3:
            if particles.shape[1:] != (n_hemispheres, component_dim):
                raise ValueError("Dimension is not correct.")
            return particles

        raise ValueError(
            "Dirac locations must have shape (n, total_dim) or "
            "(n, n_hemispheres, dim_hemisphere + 1)."
        )

    def as_component_array(self):
        """Return locations with shape ``(n, n_hemispheres, dim_hemisphere + 1)``."""
        return self._as_component_array(self.d, self.dim_hemisphere, self.n_hemispheres)

    def as_flat_array(self):
        """Return locations with shape ``(n, (dim_hemisphere + 1) * n_hemispheres)``."""
        return reshape(self.as_component_array(), (self.d.shape[0], self.input_dim))

    def _normalize_component_index(self, component_index):
        """Return a non-boolean scalar component index in range."""
        try:
            index_array = array(component_index)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("component_index must be a scalar integer.") from exc

        if ndim(index_array) != 0:
            raise ValueError("component_index must be a scalar integer.")
        if is_bool(index_array):
            raise ValueError("component_index must be an integer, not a boolean.")

        try:
            normalized_index = operator_index(component_index)
        except TypeError:
            try:
                normalized_index = operator_index(index_array.item())
            except (AttributeError, TypeError) as exc:
                raise ValueError("component_index must be an integer.") from exc

        normalized_index = int(normalized_index)
        if normalized_index < 0 or normalized_index >= self.n_hemispheres:
            raise ValueError("component_index is out of range.")
        return normalized_index

    def component_particles(self, component_index):
        """Return Dirac locations for one hyperhemisphere component."""
        component_index = self._normalize_component_index(component_index)
        return self.as_component_array()[:, component_index, :]

    def get_manifold_size(self):
        hemisphere_size = (
            0.5
            * AbstractHypersphereSubsetDistribution.compute_unit_hypersphere_surface(
                self.dim_hemisphere
            )
        )
        return hemisphere_size**self.n_hemispheres

    def moment(self, component_index=None):
        """Return weighted second moments for the product components."""
        if component_index is not None:
            return self._moment_for_component(component_index)
        return stack(
            [self._moment_for_component(i) for i in range(self.n_hemispheres)], 0
        )

    def _moment_for_component(self, component_index):
        particles = self.component_particles(component_index)
        weighted_outer_products = stack(
            [
                self.w[i] * outer(particles[i, :], particles[i, :])
                for i in range(self.d.shape[0])
            ],
            0,
        )
        return sum(weighted_outer_products, axis=0) / sum(self.w)

    def mean_axis(self, component_index=None):
        """Return weighted principal axes for the product components."""
        if component_index is not None:
            return self._mean_axis_for_component(component_index)
        return stack(
            [self._mean_axis_for_component(i) for i in range(self.n_hemispheres)], 0
        )

    def _mean_axis_for_component(self, component_index):
        eigenvalues, eigenvectors = linalg.eigh(
            self._moment_for_component(component_index)
        )
        axis = eigenvectors[:, argmax(eigenvalues)]
        axis = axis / linalg.norm(axis)
        return where(axis[-1:] < 0.0, -axis, axis)

    def mean(self):
        return self.mean_axis()

    def apply_function_component_wise(
        self, f: Callable, f_supports_multiple: bool = True
    ):
        """
        Apply a function to the Dirac locations and return a new distribution.

        :param f: Function to apply.
        :returns: A new distribution with the function applied to the locations.
        """
        if not f_supports_multiple:
            raise ValueError("Function must support multiple inputs.")
        dist = copy.deepcopy(self)
        expected_shape = (self.d.shape[0], self.component_dim)
        transformed_components = []
        for component_index in range(self.n_hemispheres):
            transformed_component = array(f(self.component_particles(component_index)))
            if transformed_component.shape != expected_shape:
                raise ValueError(
                    "Function must preserve particle count and component dimension; "
                    f"expected shape {expected_shape}, got "
                    f"{transformed_component.shape}."
                )
            transformed_components.append(transformed_component)

        component_values = stack(transformed_components, 1)
        if self._store_flat:
            dist.d = reshape(component_values, (self.d.shape[0], self.input_dim))
        else:
            dist.d = component_values
        return dist
