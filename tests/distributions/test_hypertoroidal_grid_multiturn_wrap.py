import numpy.testing as npt
from pyrecest.backend import array, pi
from pyrecest.distributions.hypertorus.hypertoroidal_grid_distribution import (
    HypertoroidalGridDistribution,
)


def test_custom_grid_nearest_neighbor_is_invariant_to_full_turns():
    grid = array([[0.0], [pi]])
    distribution = HypertoroidalGridDistribution(array([3.0, 1.0]), grid=grid)

    base_queries = array([[0.1], [pi + 0.1]])
    shifted_queries = base_queries + array([[4.0 * pi], [-4.0 * pi]])

    npt.assert_allclose(
        distribution.value_of_closest(shifted_queries),
        distribution.value_of_closest(base_queries),
    )
    npt.assert_allclose(
        distribution.pdf(shifted_queries),
        distribution.pdf(base_queries),
    )
    npt.assert_allclose(
        distribution.get_closest_point(shifted_queries[0]),
        grid[0],
    )
    npt.assert_allclose(
        distribution.get_closest_point(shifted_queries[1]),
        grid[1],
    )
