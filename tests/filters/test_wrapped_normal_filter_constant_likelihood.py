import numpy.testing as npt
from pyrecest.backend import array
from pyrecest.distributions import WrappedNormalDistribution
from pyrecest.filters.wrapped_normal_filter import WrappedNormalFilter


def _constant_positive_likelihood(_z, _x):
    return 3.0


def test_update_nonlinear_progressive_constant_likelihood_is_noop():
    filt = WrappedNormalFilter(WrappedNormalDistribution(array(0.4), array(0.7)))
    initial_mu = filt.filter_state.mu
    initial_sigma = filt.filter_state.sigma

    filt.update_nonlinear_progressive(_constant_positive_likelihood, 0.1)

    npt.assert_allclose(filt.filter_state.mu, initial_mu, rtol=0.0, atol=0.0)
    npt.assert_allclose(filt.filter_state.sigma, initial_sigma, rtol=0.0, atol=0.0)
