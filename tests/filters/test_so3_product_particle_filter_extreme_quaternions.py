import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, to_numpy
from pyrecest.filters import SO3ProductParticleFilter


def test_extreme_finite_quaternions_normalize_without_overflow():
    backend_dtype = to_numpy(array([1.0])).dtype
    largest = np.finfo(backend_dtype).max
    particles = array([[[largest / 2.0, largest / 2.0, 0.0, 0.0]]])

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        filt = SO3ProductParticleFilter(
            n_particles=1,
            num_rotations=1,
            initial_particles=particles,
        )

    expected = np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0, 0.0])
    actual = to_numpy(filt.particles[0, 0])
    npt.assert_allclose(actual, expected, rtol=1e-6, atol=0.0)
    npt.assert_allclose(np.linalg.norm(actual), 1.0, rtol=1e-6)
