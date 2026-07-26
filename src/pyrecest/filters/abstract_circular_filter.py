"""Abstract base class for circular filters."""

# pylint: disable=no-name-in-module,no-member
from math import isfinite

from pyrecest.backend import array, asarray, pi
from scipy.integrate import quad

from .abstract_filter import AbstractFilter
from .manifold_mixins import CircularFilterMixin


def _scalar_pdf_value(value, owner: str):
    """Return one finite real PDF value and reject ambiguous outputs."""
    flattened = asarray(value).reshape(-1)
    if flattened.shape[0] != 1:
        raise ValueError(
            f"{owner}.pdf must return exactly one value for one integration angle."
        )
    try:
        scalar = float(flattened[0])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{owner}.pdf must return a finite real value for one integration angle."
        ) from exc
    if not isfinite(scalar):
        raise ValueError(
            f"{owner}.pdf must return a finite real value for one integration angle."
        )
    return scalar


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
            estimate_pdf = _scalar_pdf_value(
                estimate.pdf(angle_array),
                "filter_state",
            )
            likelihood_pdf = _scalar_pdf_value(
                likelihood.pdf(angle_array),
                "likelihood",
            )
            return estimate_pdf * likelihood_pdf

        likelihood_val, _ = quad(integrand, 0.0, 2.0 * pi)
        return likelihood_val
