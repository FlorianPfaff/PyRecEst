from pyrecest.backend import empty
from pyrecest.distributions.hypersphere_subset.abstract_hyperhemispherical_distribution import (
    AbstractHyperhemisphericalDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_dirac_distribution import (
    HyperhemisphericalDiracDistribution,
)

from .abstract_particle_filter import AbstractParticleFilter
from .manifold_mixins import HyperhemisphericalFilterMixin


class HyperhemisphericalParticleFilter(
    AbstractParticleFilter, HyperhemisphericalFilterMixin
):
    def __init__(self, n_particles: int, dim: int) -> None:
        """
        Constructor

        Parameters:
        n_particles (int > 0): Number of particles to use
        dim (int > 0): Dimension
        """
        initial_filter_state = HyperhemisphericalDiracDistribution(
            empty((n_particles, dim + 1))
        )
        HyperhemisphericalFilterMixin.__init__(self)
        AbstractParticleFilter.__init__(self, initial_filter_state=initial_filter_state)

    def set_state(self, new_state):
        """
        Sets the current system state

        Parameters:
        dist_ (HyperhemisphericalDiracDistribution): New state
        """
        assert isinstance(new_state, AbstractHyperhemisphericalDistribution)
        if not isinstance(new_state, HyperhemisphericalDiracDistribution):
            new_state = HyperhemisphericalDiracDistribution(
                new_state.sample(self.filter_state.d.shape[0])
            )
        self.filter_state = new_state
