import numpy.testing as npt
from pyrecest.backend import array, pi
from pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution import (
    HypertoroidalGridDistribution,
)


def test_get_closest_point_returns_one_point_per_batched_query():
    grid = array(
        [
            [0.0, 0.0],
            [0.0, pi],
            [pi, 0.0],
            [pi, pi],
        ]
    )
    grid_values = array([1.0, 1.0, 1.0, 1.0]) / ((2.0 * pi) ** 2)
    distribution = HypertoroidalGridDistribution(grid_values, grid=grid)

    queries = array(
        [
            [2.0 * pi - 0.1, 0.05],
            [pi + 0.1, pi - 0.05],
        ]
    )

    closest = distribution.get_closest_point(queries)

    npt.assert_allclose(closest, array([[0.0, 0.0], [pi, pi]]))
