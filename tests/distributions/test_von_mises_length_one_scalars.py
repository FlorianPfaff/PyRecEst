import numpy as np

from pyrecest.backend import array, to_numpy
from pyrecest.distributions import VonMisesDistribution


def test_set_mean_accepts_length_one_backend_array():
    dist = VonMisesDistribution(array(0.0), array(2.0))

    shifted = dist.set_mean(array([1.0]))
    density_at_mode = np.asarray(to_numpy(shifted.pdf(array([1.0]))))

    assert np.all(np.isfinite(density_at_mode))
    assert np.all(density_at_mode > 0.0)


def test_sample_accepts_length_one_parameter_arrays():
    dist = VonMisesDistribution(array([0.3]), array([2.0]))

    samples = dist.sample(3)

    assert samples.shape == (3,)
