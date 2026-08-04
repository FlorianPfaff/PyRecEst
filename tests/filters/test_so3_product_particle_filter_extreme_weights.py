import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, to_numpy
from pyrecest.filters import SO3ProductParticleFilter


def _extreme_finite_weights():
    backend_dtype = to_numpy(array([1.0])).dtype
    largest = np.finfo(backend_dtype).max
    return array([largest, largest / 2.0, 0.0])


def test_constructor_normalizes_extreme_finite_weights():
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        filt = SO3ProductParticleFilter(
            n_particles=3,
            num_rotations=1,
            weights=_extreme_finite_weights(),
        )

    npt.assert_allclose(
        to_numpy(filt.weights),
        np.array([2.0 / 3.0, 1.0 / 3.0, 0.0]),
        rtol=1e-6,
        atol=0.0,
    )


def test_set_particles_normalizes_extreme_finite_weights():
    filt = SO3ProductParticleFilter(n_particles=3, num_rotations=1)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        filt.set_particles(filt.particles, weights=_extreme_finite_weights())

    npt.assert_allclose(
        to_numpy(filt.weights),
        np.array([2.0 / 3.0, 1.0 / 3.0, 0.0]),
        rtol=1e-6,
        atol=0.0,
    )
