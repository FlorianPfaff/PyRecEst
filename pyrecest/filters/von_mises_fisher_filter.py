# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, ndim
from pyrecest.distributions import VonMisesFisherDistribution

from .abstract_filter import AbstractFilter
from .manifold_mixins import HypersphericalFilterMixin


class VonMisesFisherFilter(AbstractFilter, HypersphericalFilterMixin):
    def __init__(self):
        HypersphericalFilterMixin.__init__(self)
        AbstractFilter.__init__(
            self, VonMisesFisherDistribution(array([1.0, 0.0]), 1.0)
        )

    @property
    def filter_state(self):
        return self._filter_state

    @filter_state.setter
    def filter_state(self, filter_state):
        assert isinstance(
            filter_state, VonMisesFisherDistribution
        ), "filter_state must be an instance of VonMisesFisherDistribution."
        self._filter_state = filter_state

    def set_state(self, state):
        """Set the filter state."""
        self.filter_state = state

    def get_estimate_mean(self):
        """Return the mean direction of the current filter state."""
        return self.filter_state.mean_direction()

    def predict_identity(self, sys_noise):
        """
        State prediction via mulitiplication. Provide zonal density for update
        Could add support for a rotation Q
        """
        assert isinstance(
            sys_noise, VonMisesFisherDistribution
        ), "sys_noise must be an instance of VonMisesFisherDistribution."
        self.filter_state = self.filter_state.convolve(sys_noise)

    def update_identity(self, meas_noise, z):
        """
        State update via mulitiplication. Provide zonal density for update
        Could add support for a rotation Q
        """
        assert isinstance(
            meas_noise, VonMisesFisherDistribution
        ), "meas_noise must be an instance of VonMisesFisherDistribution."
        assert (
            meas_noise.mu[-1] == 1
        ), "Set mu of meas_noise to [0,0,...,1] to acknowledge that the mean is discarded."
        assert (
            z.shape[0] == self.filter_state.input_dim
        ), "Dimension mismatch between measurement and state."
        assert ndim(z) == 1, "z should be a vector."
        meas_noise.mu = z
        self.filter_state = self.filter_state.multiply(meas_noise)
