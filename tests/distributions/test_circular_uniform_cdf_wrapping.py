import numpy as np
import numpy.testing as npt
import pytest
from pyrecest.backend import array, pi, to_numpy
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
    npt.assert_allclose(
        dist.cdf(shifted_x, starting_point=array(0.5 * pi)),
        array([0.0, 0.0, 0.75, 0.25]),
        atol=1e-12,
    )


def test_circular_uniform_cdf_avoids_overflow_before_wrapping():
    dist = CircularUniformDistribution()
    evaluation_points = array([1.0e308, -1.0e308])
    starting_point = -1.0e308

    with np.errstate(over="raise", invalid="raise"):
        actual = np.asarray(
            to_numpy(dist.cdf(evaluation_points, starting_point=starting_point))
        )

    period = 2.0 * np.pi
    expected = (
        np.mod(
            np.mod(np.array([1.0e308, -1.0e308]), period)
            - np.mod(starting_point, period),
            period,
        )
        / period
    )
    npt.assert_allclose(actual, expected)
    assert np.all(np.isfinite(actual))
    assert np.all((actual >= 0.0) & (actual < 1.0))


@pytest.mark.parametrize(
    "starting_point",
    (
        [0.0],
        np.array([0.0, 1.0]),
        True,
        float("nan"),
        float("inf"),
        1.0 + 1.0j,
        "0.0",
    ),
)
def test_circular_uniform_cdf_rejects_invalid_starting_points(starting_point):
    dist = CircularUniformDistribution()

    with pytest.raises(ValueError, match="starting_point.*finite real scalar"):
        dist.cdf(array([0.0]), starting_point=starting_point)


@pytest.mark.parametrize(
    "evaluation_points",
    (
        np.array([True]),
        np.array([float("nan")]),
        np.array([float("inf")]),
        np.array([float("-inf")]),
        np.array([1.0 + 1.0j]),
        np.array(["0.0"]),
    ),
)
def test_circular_uniform_cdf_rejects_invalid_evaluation_points(evaluation_points):
    dist = CircularUniformDistribution()

    with pytest.raises(ValueError, match="xa.*finite real"):
        dist.cdf(evaluation_points)
