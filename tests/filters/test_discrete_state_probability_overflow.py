import unittest

import numpy as np
from pyrecest.filters.discrete_state import discrete_forward_backward


class TestDiscreteStateProbabilityOverflow(unittest.TestCase):
    def test_large_finite_initial_probabilities_keep_relative_mass(self):
        largest = np.finfo(float).max
        initial_probabilities = np.array([largest, largest / 2.0])

        result = discrete_forward_backward(
            np.zeros((1, 2)),
            np.eye(2),
            initial_probabilities=initial_probabilities,
        )

        np.testing.assert_allclose(
            result.filtered_probabilities[0],
            np.array([2.0 / 3.0, 1.0 / 3.0]),
        )


if __name__ == "__main__":
    unittest.main()
