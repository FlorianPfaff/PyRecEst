import copy
from numbers import Integral

from beartype import beartype

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import empty, ones
from pyrecest.distributions import AbstractHypersphericalDistribution
from pyrecest.distributions.cart_prod.hyperhemisphere_cart_prod_dirac_distribution import (
    HyperhemisphereCartProdDiracDistribution,
)
from pyrecest.distributions.hypersphere_subset.abstract_hyperhemispherical_distribution import (
    AbstractHyperhemisphericalDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_watson_distribution import (
    HyperhemisphericalWatsonDistribution,
)
from pyrecest.distributions.hypersphere_subset.von_mises_fisher_distribution import (
    VonMisesFisherDistribution,
)

from .abstract_particle_filter import AbstractParticleFilter


def _validate_positive_integer(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a positive integer.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


class HyperhemisphereCartProdParticleFilter(AbstractParticleFilter):
    def __init__(
        self, n_particles: int, dim_hemisphere: int, n_hemispheres: int
    ) -> None:
        """
        Constructor

        Parameters:
        n_particles (int > 0): Number of particles to use
        dim (int > 0): Dimension
        """
        n_particles = _validate_positive_integer(n_particles, "n_particles")
        dim_hemisphere = _validate_positive_integer(dim_hemisphere, "dim_hemisphere")
        n_hemispheres = _validate_positive_integer(n_hemispheres, "n_hemispheres")
        initial_filter_state = HyperhemisphereCartProdDiracDistribution(
            empty((n_particles, (dim_hemisphere + 1) * n_hemispheres)),
            ones(n_particles) / n_particles,
            dim_hemisphere,
            n_hemispheres,
        )
        AbstractParticleFilter.__init__(self, initial_filter_state=initial_filter_state)

    def set_state(self, new_state):
        """
        Sets the current system state

        Parameters:
        dist_ (HyperhemisphericalDiracDistribution): New state
        """
        if not isinstance(
            new_state,
            (
                AbstractHyperhemisphericalDistribution,
                HyperhemisphereCartProdDiracDistribution,
            ),
        ):
            raise TypeError(
                "new_state must be an AbstractHyperhemisphericalDistribution "
                "or HyperhemisphereCartProdDiracDistribution."
            )
        if isinstance(new_state, HyperhemisphereCartProdDiracDistribution):
            if (
                new_state.dim_hemisphere != self.filter_state.dim_hemisphere
                or new_state.n_hemispheres != self.filter_state.n_hemispheres
            ):
                raise ValueError(
                    "new_state hemisphere topology must match the filter state."
                )
            self.filter_state = new_state
            return
        if not isinstance(new_state, HyperhemisphereCartProdDiracDistribution):
            new_state = HyperhemisphereCartProdDiracDistribution(
                new_state.sample(self.filter_state.d.shape[0]),
                w=ones(self.filter_state.d.shape[0]) / self.filter_state.d.shape[0],
                dim_hemisphere=self.filter_state.dim_hemisphere,
                n_hemispheres=self.filter_state.n_hemispheres,
            )
        self.filter_state = new_state

    @beartype
    def predict_nonlinear_each_part(
        self,
        f,
        # Limit noise distribution to ones that support set_mode
        noise_distribution: (
            HyperhemisphericalWatsonDistribution | VonMisesFisherDistribution | None
        ) = None,
        function_is_vectorized: bool = True,
        shift_instead_of_add: bool = True,
    ):
        """
        Predicts the next state for each hyperhemisphere
        """
        if not function_is_vectorized:
            raise ValueError("Only vectorized functions are supported.")
        if not (
            noise_distribution is None
            or noise_distribution.dim == self.filter_state.dim_hemisphere
        ):
            raise ValueError(
                "Noise dimension must match state dimension in Cartesian product."
            )
        if not shift_instead_of_add:
            raise ValueError("Only shifting is supported.")
        if noise_distribution is None:
            self.filter_state = self.filter_state.apply_function_component_wise(f)
            return
        for i in range(self.filter_state.n_hemispheres):
            # Apply the function to each hyperhemisphere
            index_arr = range(
                i * (self.filter_state.dim_hemisphere + 1),
                (i + 1) * (self.filter_state.dim_hemisphere + 1),
            )
            # Consider only part of the state of the current hemisphere
            curr_d = self.filter_state.d[:, index_arr]
            d_fun_applied = f(curr_d)
            # Add noise
            if noise_distribution is not None:
                for j in range(self.filter_state.d.shape[0]):
                    # Set mode to transformed state
                    # This will fail if set_mode is unavailable
                    noise_curr = copy.deepcopy(noise_distribution)
                    shifted_noise = noise_curr.set_mode(d_fun_applied[j])
                    if shifted_noise is not None:
                        noise_curr = shifted_noise
                    # Sample one noise vector centered at the transformed state to add noise
                    self.filter_state.d[j, index_arr] = noise_curr.sample(1)

    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        if isinstance(
            new_state,
            (
                AbstractHyperhemisphericalDistribution,
                AbstractHypersphericalDistribution,
            ),
        ):
            if new_state.dim != self.filter_state.dim_hemisphere:
                raise ValueError(
                    "new_state dimension must match the filter state's "
                    "hemisphere dimension."
                )
            samples = new_state.sample(
                self._filter_state.d.shape[0] * self._filter_state.n_hemispheres
            )
            if isinstance(new_state, AbstractHypersphericalDistribution):
                samples[samples[:, -1] < 0] = -samples[samples[:, -1] < 0]
            self._filter_state.d = samples.reshape(self.filter_state.d.shape)
        else:
            AbstractParticleFilter.filter_state.fset(self, new_state)
