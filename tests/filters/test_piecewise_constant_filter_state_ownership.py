import unittest

import numpy.testing as npt

from pyrecest.backend import array, copy
from pyrecest.distributions.circle.piecewise_constant_distribution import (
    PiecewiseConstantDistribution,
)
from pyrecest.filters.piecewise_constant_filter import PiecewiseConstantFilter


class TestPiecewiseConstantFilterStateOwnership(unittest.TestCase):
    def test_setting_state_does_not_alias_input_distribution(self):
        filt = PiecewiseConstantFilter(4)
        state = PiecewiseConstantDistribution(array([1.0, 2.0, 3.0, 4.0]))

        filt.filter_state = state
        assigned_weights = copy(filt.filter_state.w)

        self.assertIsNot(filt.filter_state, state)
        state.w = array([4.0, 3.0, 2.0, 1.0])

        npt.assert_allclose(filt.filter_state.w, assigned_weights)


if __name__ == "__main__":
    unittest.main()
