import copy
import unittest

import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.interacting_multiple_model_filter import (
    InteractingMultipleModelFilter,
)


class StateOnlyGaussianFilter:
    def __init__(self, initial_state):
        self.filter_state = copy.deepcopy(initial_state)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Only supported on numpy backend",
)
class ImmFilterStateTransactionTest(unittest.TestCase):
    def test_invalid_replacement_bank_preserves_current_bank(self):
        imm = InteractingMultipleModelFilter(
            [
                StateOnlyGaussianFilter(
                    GaussianDistribution(array([0.0]), array([[1.0]]))
                ),
                StateOnlyGaussianFilter(
                    GaussianDistribution(array([1.0]), array([[2.0]]))
                ),
            ],
            transition_matrix=eye(2),
        )
        previous_bank = list(imm.filter_bank)
        previous_states = [
            copy.deepcopy(curr_filter.filter_state) for curr_filter in imm.filter_bank
        ]
        invalid_bank = [
            StateOnlyGaussianFilter(GaussianDistribution(array([2.0]), array([[1.0]]))),
            StateOnlyGaussianFilter(GaussianDistribution(array([3.0, 4.0]), eye(2))),
        ]

        with self.assertRaisesRegex(ValueError, "same state dimension"):
            imm.filter_state = invalid_bank

        for curr_filter, previous_filter, previous_state in zip(
            imm.filter_bank, previous_bank, previous_states
        ):
            self.assertIs(curr_filter, previous_filter)
            npt.assert_allclose(curr_filter.filter_state.mu, previous_state.mu)
            npt.assert_allclose(curr_filter.filter_state.C, previous_state.C)


if __name__ == "__main__":
    unittest.main()
