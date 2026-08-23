import numpy.testing as npt
from pyrecest.backend import array
from pyrecest.distributions.circle.circular_grid_distribution import (
    CircularGridDistribution,
)
from pyrecest.filters.abstract_grid_filter import AbstractGridFilter


def test_filter_state_assignment_does_not_alias_caller_distribution():
    grid_filter = AbstractGridFilter(CircularGridDistribution(array([1.0, 1.0, 1.0])))
    assigned_state = CircularGridDistribution(array([1.0, 2.0, 3.0]))

    grid_filter.filter_state = assigned_state

    assert grid_filter.filter_state is not assigned_state
    assigned_state.grid_values = array([9.0, 9.0, 9.0])
    npt.assert_allclose(grid_filter.filter_state.grid_values, array([1.0, 2.0, 3.0]))
