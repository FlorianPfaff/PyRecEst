import copy
import unittest

import numpy.testing as npt
import pyrecest.backend

from pyrecest.filters.hyperhemispherical_grid_filter import (
    HyperhemisphericalGridFilter,
)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
    reason="Not supported on JAX backend",
)
class TestHyperhemisphericalGridFilterStateOwnership(unittest.TestCase):
    def test_filter_state_assignment_does_not_alias_caller_distribution(self):
        grid_filter = HyperhemisphericalGridFilter(50, 2)
        assigned_state = copy.deepcopy(grid_filter.filter_state)
        expected_values = copy.deepcopy(assigned_state.grid_values)

        grid_filter.filter_state = assigned_state

        self.assertIsNot(grid_filter.filter_state, assigned_state)
        assigned_state.grid_values = assigned_state.grid_values * 2.0
        npt.assert_allclose(grid_filter.filter_state.grid_values, expected_values)
