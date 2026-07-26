import numpy.testing as npt
from pyrecest.backend import array, to_numpy
from pyrecest.distributions import (
    HyperhemisphericalGridDistribution,
    SO3DiracDistribution,
)


def test_so3_dirac_conversion_reuses_hyperhemispherical_grid_without_particle_count():
    grid = array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    grid_values = array([1.0, 3.0])
    source = HyperhemisphericalGridDistribution(grid, grid_values)

    converted = SO3DiracDistribution.from_distribution(source)

    npt.assert_allclose(to_numpy(converted.d), to_numpy(grid))
    npt.assert_allclose(to_numpy(converted.w), [0.25, 0.75])
