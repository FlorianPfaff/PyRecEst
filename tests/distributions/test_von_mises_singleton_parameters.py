import numpy as np
from pyrecest.backend import to_numpy
from pyrecest.distributions import VonMisesDistribution


def _numpy_shape(value):
    return np.asarray(to_numpy(value)).shape


def test_constructor_normalizes_singleton_sequence_parameters():
    distribution = VonMisesDistribution([0.3], [2.0], norm_const=[1.5])

    assert _numpy_shape(distribution.mu) == ()
    assert _numpy_shape(distribution.kappa) == ()
    assert _numpy_shape(distribution.norm_const) == ()

    samples = np.asarray(to_numpy(distribution.sample(5)))
    assert samples.shape == (5,)
    assert np.all(np.isfinite(samples))


def test_set_mean_normalizes_singleton_sequence():
    distribution = VonMisesDistribution(0.3, 2.0).set_mean([0.7])

    assert _numpy_shape(distribution.mu) == ()
    density = np.asarray(to_numpy(distribution.pdf([0.7])))
    assert density.shape == (1,)
    assert np.all(np.isfinite(density))
