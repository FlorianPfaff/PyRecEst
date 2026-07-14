import pytest

from pyrecest.distributions.circle.circular_grid_distribution import (
    CircularGridDistribution,
)


@pytest.mark.parametrize(
    "grid_values",
    (
        [],
        1.0,
        [[1.0, 2.0]],
    ),
)
def test_rejects_invalid_grid_value_shapes(grid_values):
    with pytest.raises(
        ValueError,
        match="grid_values must be a non-empty one-dimensional array",
    ):
        CircularGridDistribution(grid_values)


def test_accepts_singleton_grid_value_vector():
    distribution = CircularGridDistribution([1.0])

    assert distribution.grid_values.shape == (1,)
    assert distribution.grid.shape == (1,)
