from .lin_periodic_particle_filter import LinPeriodicParticleFilter

from pyrecest.distributions.cart_prod.hypercylindrical_dirac_distribution import HypercylindricalDiracDistribution
import copy
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import mod, pi, zeros

class HypercylindricalParticleFilter(LinPeriodicParticleFilter):
    def __init__(self, n_particles, bound_dim, lin_dim):
        """
        Constructor

        Parameters:
        n_particles (int > 0): Number of particles to use
        bound_d (int >= 0): Number of bounded dimensions
        lin_d (int >= 0): Number of linear dimensions
        """
        LinPeriodicParticleFilter.__init__(self, HypercylindricalDiracDistribution(bound_dim, zeros((n_particles, bound_dim + lin_dim))))

    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    def filter_state(self, new_state):
        """
        Sets the current system state.

        Parameters:
        new_state (AbstractHypercylindricalDistribution): new state
        """
        if not isinstance(new_state, HypercylindricalDiracDistribution):
            state_to_set = HypercylindricalDiracDistribution.from_distribution(new_state, self.filter_state.w.shape[0])
        else:
            state_to_set = copy.deepcopy(new_state)
            
        self._filter_state = state_to_set

    def predict_nonlinear(self, f, noise_distribution=None, function_is_vectorized=True):
        """
        Predicts assuming a nonlinear system model, i.e.,
        x(k+1) = f(x(k)) + w(k) with mod 2pi applied to the periodic dimensions and
        where w(k) being additive noise distributed according to noise_distribution.

        Parameters:
        f (function): system function
        noise_distribution (AbstractHypercylindricalDistribution): distribution of additive noise
        function_is_vectorized (bool): True if the function is vectorized, False otherwise
        """
        super().predict_nonlinear(f, noise_distribution, function_is_vectorized, False)
        # Wrap bounded dimensions
        self.filter_state.d[:, 0:self.filter_state.bound_dim] = mod(self.filter_state.d[:, 0:self.filter_state.bound_dim], 2 * pi) # noqa: E203