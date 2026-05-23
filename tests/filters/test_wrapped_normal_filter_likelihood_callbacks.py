import math

import numpy as np
from pyrecest.backend import array, random
from pyrecest.distributions import WrappedNormalDistribution
from pyrecest.filters import WrappedNormalFilter


def _scalar_only_likelihood(z, x):
    return math.exp(math.cos(float(z) - float(x)))


def _make_filter():
    return WrappedNormalFilter(WrappedNormalDistribution(array(0.0), array(0.5)))


def test_update_nonlinear_particle_accepts_scalar_likelihood_callback():
    random.seed(0)
    filt = _make_filter()

    filt.update_nonlinear_particle(_scalar_only_likelihood, 0.1)

    assert isinstance(filt.filter_state, WrappedNormalDistribution)
    assert np.isfinite(float(filt.filter_state.sigma))


def test_update_nonlinear_progressive_accepts_scalar_likelihood_callback():
    filt = _make_filter()

    filt.update_nonlinear_progressive(_scalar_only_likelihood, 0.1)

    assert isinstance(filt.filter_state, WrappedNormalDistribution)
    assert np.isfinite(float(filt.filter_state.sigma))
