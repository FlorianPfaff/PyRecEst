"""Abstract base class for circular filters."""

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, asarray, pi
from scipy.integrate import quad

from .abstract_filter import AbstractFilter
from .manifold_mixins import CircularFilterMixin


class AbstractCircularFilter(AbstractFilter, CircularFilterMixin):
    """Abstract base class for filters on S1/SO(2)."""

    def __init__(self, initial_filter_state):
        CircularFilterMixin.__init__(self)
        AbstractFilter.__init__(self, initial_filter_state)

    def get_estimate(self):
        """Return the current filter state."""
        return self.filter_state

    def association_likelihood(self, likelihood):
        """Return the association likelihood."""
        return self.association_likelihood_numerical(likelihood)

    def association_likelihood_numerical(self, likelihood):
        """Numerically integrate estimate times likelihood over [0, 2*pi)."""
        estimate = self.get_estimate()

        def integrand(angle):
            angle_array = array([angle])
            estimate_pdf = asarray(estimate.pdf(angle_array)).reshape(-1)[0]
            likelihood_pdf = asarray(likelihood.pdf(angle_array)).reshape(-1)[0]
            return float(estimate_pdf * likelihood_pdf)

        likelihood_val, _ = quad(integrand, 0.0, 2.0 * pi)
        return likelihood_val
