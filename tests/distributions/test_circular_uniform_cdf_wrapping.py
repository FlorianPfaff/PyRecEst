import numpy.testing as npt
from pyrecest.backend import array, pi
from pyrecest.distributions.circle.circular_uniform_distribution import (
    CircularUniformDistribution,
)


def test_circular_uniform_cdf_wraps_angles_relative_to_starting_point():
    dist = CircularUniformDistribution()

    x = array([0.0, 2.0 * pi, -pi, 3.0 * pi])
    npt.assert_allclose(dist.cdf(x), array([0.0, 0.0, 0.5, 0.5]), atol=1e-12)

    shifted_x = array([0.5 * pi, 2.5 * pi, 0.0, pi])
    npt.assert_allclose(
        dist.cdf(shifted_x, starting_point=0.5 * pi),
        array([0.0, 0.0, 0.75, 0.25]),
        atol=1e-12,
    )
